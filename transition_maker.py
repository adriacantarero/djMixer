import numpy as np
from feature_extractor import TrackFeatures
from pydub import AudioSegment
from pydub.effects import high_pass_filter
from dataclasses import dataclass
import pyrubberband as pyrb # type: ignore
import math

@dataclass
class Transition:
    from_track: TrackFeatures  # track transitioning out
    to_track: TrackFeatures    # track transitioning in
    fade_out_start: int        # Start timestamp of the fade-out for track 1 (ms)
    fade_out_end: int          # End timestamp of the fade-out for track 1 (ms)
    fade_in_start: int         # Start timestamp of the fade-in for track 2 (ms)
    fade_in_end: int           # End timestamp of the fade-in for track 2 (ms)

class TransitionMaker:
    """
    Calculates the optimal transition points and renders a mix with dynamic transition types.
    """
    def __init__(self, transition_length_beats: int, cost_weights: dict):
        """
        Initializes the maker with algorithm parameters like transition length and cost weights.

        Parameters
        ----------
        transition_length_beats (int): Number of beats that a transition should span.
        cost_weights (dict): Dictionary of weights used to compute the transition cost between tracks.
        """
        self.transition_length_beats = transition_length_beats
        self.cost_weights = cost_weights

    # ============================
    # TRANSITION POINT SELECTION =
    # ============================
    def _time_to_beat_index(self, times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
        """
        Convert an array of times (in seconds) to the indices of the closest beats.

        Parameters
        ----------
        times (np.ndarray): time points that need to be mapped to beat indices.
        beat_times (np.ndarray): beat times corresponding to the track.

        Returns
        -------
        np.ndarray: indices of the closest beats for each input time.
        """
        return np.array([np.argmin(np.abs(beat_times - t)) for t in times])
    
    def find_best_transition(self, track1: TrackFeatures, track2: TrackFeatures) -> tuple:
        """
        Method to find the best transition between two tracks (spotify-based).

        Parameters
        ----------
        track1 (TrackFeatures): outgoing track
        track2 (TrackFeatures): ingoing track
        
        Returns
        -------
        tuple: pair of times representing the best transition points in track1 and track2.
        """
        
        # Get transition candidates
        t_1 = self._get_pruned_candidates(track1, is_outgoing=True)
        t_2 = self._get_pruned_candidates(track2, is_outgoing=False)
        
        # Calculate costs of potential transitions
        cost_matrices = self._calculate_all_cost_matrices(
            track1=track1,
            track2=track2,
            candidates1=t_1,
            candidates2=t_2
        )

        # Find best transition times
        Lambda = self._combine_costs(cost_matrices=cost_matrices)
        t1_start, t2_start = np.unravel_index(np.argmin(Lambda), Lambda.shape)

        # Compute end points of the transition
        t1_end = t1_start + self.transition_length_beats
        t2_end = t2_start + self.transition_length_beats
        
        return (track1.beat_times[t1_start], track1.beat_times[t1_end]), (track2.beat_times[t2_start], track2.beat_times[t2_end])

    def _get_pruned_candidates(self, track: TrackFeatures, is_outgoing: bool) -> np.ndarray:
        """
        Identifies and prunes the possible start points (downbeats) for a transition.

        Parameters
        ----------
        track (TrackFeatures): The track containing beat information.
        is_outgoing (bool): Whether the transition is outgoing (True) or incoming (False).
        """
        # Compute downbeat indices
        downbeat_indices = self._time_to_beat_index(track.downbeat_times, track.beat_times)

        # Exclude the last 'transition_length_beats' beats
        track_length = track.beat_times[-1]
        beats = track.beat_times[:-self.transition_length_beats]
        
        # Keep only downbeat indices that are within the valid beat range
        valid_downbeat_indices = downbeat_indices[downbeat_indices < len(beats)]
        downbeats = beats[valid_downbeat_indices]
        
        if is_outgoing:
            # Consider downbeats that are in the last 25% of the track
            return valid_downbeat_indices[np.where(downbeats >= track_length*0.75)]
        
        # Consider downbeats that are in the first 20% of the track
        return valid_downbeat_indices[np.where(downbeats <= track_length*0.20)]

    def _calculate_all_cost_matrices(self, track1: TrackFeatures, track2: TrackFeatures, candidates1: np.ndarray, candidates2: np.ndarray) -> dict:
        """
        Computes cost matrices (timbre, chroma, etc.) for all candidate pairs.
        """
        # Precompute section boundaries for both tracks
        seg_boundaries_1 = self._time_to_beat_index(track1.segment_times, track1.beat_times)
        seg_boundaries_2 = self._time_to_beat_index(track2.segment_times, track2.beat_times)

        # Total number of beats in each track
        n = len(track1.beat_times)
        m = len(track2.beat_times)

        # Initialize cost matrices with infinity
        Lambda_T = np.ones((n,m))*np.inf # Timbre cost
        Lambda_C = np.copy(Lambda_T)     # Chroma cost
        Lambda_S = np.copy(Lambda_T)     # Structural cost
        Lambda_l = np.copy(Lambda_T)     # Loudness cost
        #Lambda_v = np.copy(Lambda_T) # Vocalness cost
        
        # Compute pair-wise comparisons along the features over overlapping regions (transitions)
        for i in candidates1:
            for j in candidates2:
                i_n = i + self.transition_length_beats
                j_n = j + self.transition_length_beats
                
                Lambda_T[i, j] = np.linalg.norm(track1.mfcc_features[:, i:i_n] - track2.mfcc_features[:, j:j_n])
                Lambda_C[i, j] = np.linalg.norm(track1.chroma_features[:, i:i_n] - track2.chroma_features[:, j:j_n])
                Lambda_S[i, j] = float(int(i_n not in seg_boundaries_1) + int(j_n not in seg_boundaries_2))
                Lambda_l[i, j] = np.mean(2 - (track1.loudness_features[:, i:i_n] + track2.loudness_features[:, j:j_n]))
                #Lambda_v[i, j] = np.mean(track1.vocalness_features[:, i:i_n]) + np.mean(track2.vocalness_features[:, j:j_n])
        
        # Combine all computed cost matrices into a dictionary
        cost_matrices = {
            "T": Lambda_T,
            "C": Lambda_C,
            "S": Lambda_S,
            "l": Lambda_l,
            #"v": Lambda_v
        }
        return cost_matrices

    def _combine_costs(self, cost_matrices: dict) -> np.ndarray:
        """
        Standardizes and combines the cost matrices using the defined weights.

        Parameters
        ----------
        cost_matrices (dict): Dictionary of cost matrices, where keys correspond to feature types (e.g., "T", "C", "S", "l").

        Returns
        -------
        np.ndarray: The final combined cost matrix with weighted contributions from each feature.
        """
        # Init final cost matrix
        final_lambda = np.zeros_like(cost_matrices["T"])

        # Iterate over all provided cost matrices
        for key in cost_matrices:
            if key not in self.cost_weights or self.cost_weights[key] == 0:
                continue
                
            cost = cost_matrices[key]

            # Extract valid finite costs for normalization
            valid_costs = cost[cost != np.inf]
            if len(valid_costs) == 0:
                continue
            
            # Find min and max for normalization
            min_val = np.min(valid_costs)
            max_val = np.max(valid_costs)
            
            # Normalize cost matrix to [0, 1] range
            if max_val > min_val:
                cost_normalized = (cost - min_val) / (max_val - min_val)
            else:
                cost_normalized = np.zeros_like(cost)

            cost_normalized[cost == np.inf] = 1.0
            
            # Add weighted contribution of this normalized matrix to the final combined matrix
            final_lambda += self.cost_weights[key] * cost_normalized
            
        return final_lambda

    # ===================
    # RENDERING METHODS =
    # ===================
    def create_transition(self, transition: Transition, return_transition = True) -> AudioSegment:
        """
        Creates a smooth transition between two tracks, evolving from outgoing song's features (tempo, pitch) to incoming's.
        
        Parameters:
        -----------
        transition (Transition): Contains information about the two tracks to be mixed
        return_transition (bool): Whether the function returns the transition segment or the whole mix
        
        Returns:
        --------
        AudioSegment: Mixed audio with smooth transition
        """
        # Extract data
        track_out = transition.from_track
        track_in = transition.to_track

        # Read audio file w. Pydub
        try:
            audio_out = AudioSegment.from_file(track_out.track_path)
            audio_in = AudioSegment.from_file(track_in.track_path)

            # Convert to mono audio if not already
            audio_out = audio_out.set_channels(1)
            audio_in = audio_in.set_channels(1)

            # Get audio transition regions with bounds checking
            fade_out = audio_out[transition.fade_out_start:min(len(audio_out), transition.fade_out_end)]
            fade_in = audio_in[transition.fade_in_start:min(len(audio_in), transition.fade_in_end)]

            # Calculate beat positions from timing information (later use)
            out_start = np.argmin(np.abs(track_out.beat_times - (transition.fade_out_start / 1000)))
            in_start = np.argmin(np.abs(track_in.beat_times - (transition.fade_in_start / 1000)))

        except Exception as e:
            raise ValueError(f"Error loading audio files: {e}")

        # ====================================
        # Synchronize beats, keys and energy =
        # ====================================
        pitch_shift_steps = self._calculate_pitch_shift(track_out, track_in)

         # Evolve from outgoing song tempo and pitch to incoming's
        adjusted_fade_out, adjusted_fade_in = self.sync_tempo_pitch(fade_out, fade_in, pitch_shift_steps)
        
        # ===================================
        # Render the best transition choice =
        # ===================================
        transition_type = self._select_transition_type(track_out, track_in, out_start, in_start)

        # Create the crossfade with the resulting transition type
        if transition_type == "HARMONIC_BLEND":
            crossfade = self._render_harmonic_blend(adjusted_fade_out, adjusted_fade_in)
        else: # Default to STANDARD_EQ_MIX
            crossfade = self._render_standard_eq_mix(adjusted_fade_out, adjusted_fade_in)

        # Return only transition +5s from each song if requested (easy analyisis)
        if return_transition:
            outgoing_part = audio_out[transition.fade_out_start-5000:transition.fade_out_start]
            incoming_part = audio_in[transition.fade_in_end:transition.fade_in_end+10000]

            transition = outgoing_part + crossfade + incoming_part
            return transition
        
        # Combine all three parts for the final result
        outgoing_part = audio_out[:transition.fade_out_start]
        incoming_part = audio_in[transition.fade_in_end:]

        final_mix = outgoing_part + crossfade + incoming_part
        return final_mix

    def sync_tempo_pitch(self, fade_out: AudioSegment, fade_in: AudioSegment, pitch_shift_steps: float):
        """
        Apply beat-synchronized tempo and pitch evolution between two audio segments.
        
        Parameters:
        -----------
        fade_out: AudioSegment from outgoing track
        fade_in: AudioSegment from incoming track  
        pitch_shift_steps: Target pitch shift for incoming track
        
        Returns:
        --------
        tuple: (processed_fade_out, processed_fade_in) with evolved tempo and pitch
        """
        # Convert to numpy arrays for rubberband
        fade_out_arr = np.array(fade_out.get_array_of_samples(), dtype=np.float32) / (2**(fade_out.sample_width * 8 - 1))
        fade_in_arr = np.array(fade_in.get_array_of_samples(), dtype=np.float32) / (2**(fade_in.sample_width * 8 - 1))
        sr = fade_out.frame_rate
        N = self.transition_length_beats

        # Calculate beat durations in milliseconds
        beat_duration_out = len(fade_out) / N
        beat_duration_in = len(fade_in) / N
        
        out_beats_processed, in_beats_processed = [], []      
        for n in range(N):
            # Calculate target duration for this beat (tempo evolution)
            d_target = ((N - n) / N) * beat_duration_out + (n / N) * beat_duration_in
            
            # Calculate stretch factors
            stretch_out = beat_duration_out / d_target
            stretch_in = beat_duration_in / d_target
            
            # Calculate current pitch shift (pitch evolution)
            current_pitch_shift = (n / N) * pitch_shift_steps
            
            # Extract nth beat from each segment
            beat_start_out = int(n * beat_duration_out * sr / 1000)
            beat_end_out = int((n + 1) * beat_duration_out * sr / 1000)
            beat_start_in = int(n * beat_duration_in * sr / 1000) 
            beat_end_in = int((n + 1) * beat_duration_in * sr / 1000)
            
            # Ensure we don't exceed array bounds
            beat_out = fade_out_arr[beat_start_out:min(beat_end_out, len(fade_out_arr))]
            beat_in = fade_in_arr[beat_start_in:min(beat_end_in, len(fade_in_arr))]
            
            # Process beats with tempo and pitch evolution
            if len(beat_out) > 0:
                stretched_out = pyrb.time_stretch(beat_out, sr, stretch_out)
                if current_pitch_shift != 0:
                    stretched_out = pyrb.pitch_shift(stretched_out, sr, current_pitch_shift) # evolve key: out --> in 
                out_beats_processed.append(stretched_out)
            
            if len(beat_in) > 0:
                stretched_in = pyrb.time_stretch(beat_in, sr, stretch_in)
                in_beats_processed.append(stretched_in)
        
        # Concatenate and convert back to AudioSegment
        sample_width = fade_out.sample_width
        def _array_to_audio(arr):
            if len(arr) == 0:
                return AudioSegment.silent(duration=0)
            concatenated = np.concatenate(arr) if len(arr) > 1 else arr[0]
            samples = np.clip(concatenated * 32768.0, -32768, 32767).astype(np.int16)
            return AudioSegment(
                samples.tobytes(),
                frame_rate=sr,
                sample_width=sample_width,
                channels=1
            )
        
        return _array_to_audio(out_beats_processed), _array_to_audio(in_beats_processed)
    
    def _calculate_pitch_shift(self, track_out: TrackFeatures, track_in: TrackFeatures) -> float:
        """
        Calculate pitch shift needed for outgoing track to match incoming track's key.
        """
        key_diff = (track_in.key - track_out.key) % 12
        # Convert to minimal semitone shift (-6 to +5)
        if key_diff > 6:
            key_diff -= 12
        return key_diff

    def _select_transition_type(self, track1: TrackFeatures, track2: TrackFeatures, i: int = 0, j: int = 0) -> str:
        """
        Decide whether to use a harmonic blend or standard EQ mix.
        """
        HARMONIC_SIMILARITY_THRESHOLD = 0.8

        i_n = i + self.transition_length_beats
        j_n = j + self.transition_length_beats
        
        chroma_dist = np.linalg.norm(track1.chroma_features[:, i:i_n] - track2.chroma_features[:, j:j_n])
        max_possible_dist = np.sqrt(self.transition_length_beats * 12)
        harmonic_similarity = 1 - (chroma_dist / max_possible_dist)

        if harmonic_similarity > HARMONIC_SIMILARITY_THRESHOLD:
            return "HARMONIC_BLEND"
        
        return "STANDARD_EQ_MIX"
    
    def _render_harmonic_blend(self, fade_out: AudioSegment, fade_in: AudioSegment) -> AudioSegment:
        """
        Creates a smooth crossfade for harmonically compatible tracks.
        """
        blend_ms = len(fade_out)
        return fade_out.fade_out(blend_ms).overlay(fade_in.fade_in(blend_ms))

    def _render_standard_eq_mix(self, fade_out: AudioSegment, fade_in: AudioSegment) -> AudioSegment:
        """
        EQ transition with smooth bass and frequency handover.
        """
        transition_ms = len(fade_out)
        
        # 3-stage transition for smoother results
        stage1_end = transition_ms // 3
        stage2_end = 2 * transition_ms // 3
        
        mixed_segments = []
        
        # Stage 1: Outgoing track dominant, incoming track high-passed (removed bass low frequencies)
        stage1_out = fade_out[:stage1_end]
        stage1_in = high_pass_filter(fade_in[:stage1_end], 250).apply_gain(-6)  # Reduced presence
        mixed_segments.append(stage1_out.overlay(stage1_in))
        
        # Stage 2: Equal presence with bass transition
        stage2_out = high_pass_filter(fade_out[stage1_end:stage2_end], 120) # Outgoing loses bass
        stage2_in = fade_in[stage1_end:stage2_end]
        # Crossfade within this stage
        stage2_duration = stage2_end - stage1_end
        crossfade_stage2 = stage2_out.fade_out(stage2_duration).overlay(
            stage2_in.fade_in(stage2_duration)
        )
        mixed_segments.append(crossfade_stage2)
        
        # Stage 3: Incoming track dominant, outgoing track filtered out
        stage3_out = high_pass_filter(fade_out[stage2_end:], 500).apply_gain(-12)  # Fading out
        stage3_in = fade_in[stage2_end:] # Full incoming track
        mixed_segments.append(stage3_out.overlay(stage3_in))
        
        # Combine all stages, starting with empty audio
        final_mix = sum(mixed_segments, AudioSegment.silent(duration=0))
        
        return final_mix