import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from feature_extractor import TrackFeatures

class TrackSequencer:
    """
    Determines the optimal order for a list of tracks by modeling the problem
    as finding the shortest Hamiltonian Path in a graph.
    """
    def __init__(self, sequencing_weights: Dict[str, float]):
        """
        Initializes the sequencer with weights for each feature component.
        These weights control the importance of style, key, and tempo in the distance calculation.
        """
        if not np.isclose(sum(sequencing_weights.values()), 1.0):
            raise ValueError("Sequencing weights must sum to 1.0")
        self.weights = sequencing_weights

    def sequence_playlist(self, tracks: List[TrackFeatures], start_track_index: int = 0) -> List[TrackFeatures]:
        """
        Takes an unordered list of tracks and returns them in an optimal sequence.

        Args:
            tracks (List[TrackFeatures]): The list of tracks to sequence.
            start_track_index (int): The index of the track to start the sequence with.

        Returns:
            List[TrackFeatures]: The list of tracks in the calculated optimal order.
        """
        if not tracks:
            return []
        
        # Map all track features into a combined, weighted Euclidean space
        feature_vectors = self._create_feature_vectors(tracks)

        # Build the pairwise distance matrix for the complete graph
        distance_matrix = self._build_distance_matrix(feature_vectors)

        # Solve for the shortest path using the HAM-2 greedy algorithm
        optimal_indices = self._solve_shortest_path_greedy_ham2(distance_matrix, start_track_index)
        
        # Reorder the original track list based on the optimal path
        sequenced_tracks = [tracks[i] for i in optimal_indices]
        return sequenced_tracks

    def _create_feature_vectors(self, tracks: List[TrackFeatures]) -> np.ndarray:
        """
        Creates a list of combined feature vectors, one for each track.
        """
        style_vectors = np.array([t.style_vector for t in tracks])
        key_vectors = np.array([self._map_key_to_euclidean(t.key, t.mode) for t in tracks])
        tempo_vectors = np.array([self._map_tempo_to_euclidean(t.tempo) for t in tracks])

        # Scale each feature set before combining
        scaler = StandardScaler()
        style_vectors_scaled = scaler.fit_transform(style_vectors)
        key_vectors_scaled = scaler.fit_transform(key_vectors)
        tempo_vectors_scaled = scaler.fit_transform(tempo_vectors)

        all_vectors = []
        for i in range(len(tracks)):
            # Combine the scaled vectors with the weights
            combined_vector = np.concatenate([
                style_vectors_scaled[i] * np.sqrt(self.weights['style']),
                key_vectors_scaled[i] * np.sqrt(self.weights['key']),
                tempo_vectors_scaled[i] * np.sqrt(self.weights['tempo'])
            ])

            all_vectors.append(combined_vector)
            
        return np.array(all_vectors)

    def _map_key_to_euclidean(self, key: int, mode: int) -> np.ndarray:
        """
        Maps a musical key (0-11) and mode (0 for minor, 1 for major) to a 3D
        space based on the circle of fifths.
        """
        # Map key to circle of fifths angle (C=0, G=1, D=2, ...)
        # Major keys are shifted relative to minor keys
        angle_major = (key * 7) % 12
        angle_minor = (key * 7 + 9) % 12
        
        angle = angle_major if mode == 1 else angle_minor
        
        # Convert polar coordinates (angle, z) to 3D Cartesian
        radians = (angle / 12.0) * 2 * np.pi
        z = 0.5 if mode == 1 else -0.5 # Separate major and minor keys on z-axis
        x = np.cos(radians)
        y = np.sin(radians)
        
        return np.array([x, y, z])

    def _map_tempo_to_euclidean(self, tempo: float) -> np.ndarray:
        """
        Maps tempo to a 2D octave-invariant space.
        This treats tempos that are doubles/halves of each other as similar.
        """
        # Normalize tempo into the [60, 120) BPM range by doubling/halving
        while tempo < 60:
            tempo *= 2
        while tempo >= 120:
            tempo /= 2
            
        # Map the normalized tempo to an angle
        # This maps the log2 space of [60, 120) to [0, 2*pi)
        phase = np.log2(tempo / 60.0)
        radians = phase * 2 * np.pi
        
        x = np.cos(radians)
        y = np.sin(radians)

        return np.array([x, y])

    def _build_distance_matrix(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise Euclidean distance matrix for all tracks.
        The edge weight between two tracks is their distance in the combined feature space.
        """
        num_tracks = len(feature_vectors)
        dist_matrix = np.zeros((num_tracks, num_tracks))
        for i in range(num_tracks):
            for j in range(i + 1, num_tracks):
                # Calculate Euclidean distance
                dist = np.linalg.norm(feature_vectors[i] - feature_vectors[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

    def _solve_shortest_path_greedy_ham2(self, distance_matrix: np.ndarray, start_node: int) -> List[int]:
        """
        Finds an approximate solution to the Shortest Hamiltonian Path problem using
        the HAM-2 greedy algorithm. This algorithm builds a path by adding the closest 
        unvisited node to either the head or the tail of the current path.
        """
        num_tracks = len(distance_matrix)
        
        # The path starts with the given seed track
        path = [start_node]
        visited = {start_node}
        
        head = start_node
        tail = start_node

        while len(path) < num_tracks:
            # Find closest unvisited node to the current head
            head_dist = float('inf')
            head_neighbor = -1
            for i in range(num_tracks):
                if i not in visited and distance_matrix[head][i] < head_dist:
                    head_dist = distance_matrix[head][i]
                    head_neighbor = i

            # Find closest unvisited node to the current tail
            tail_dist = float('inf')
            tail_neighbor = -1
            for i in range(num_tracks):
                if i not in visited and distance_matrix[tail][i] < tail_dist:
                    tail_dist = distance_matrix[tail][i]
                    tail_neighbor = i

            # Decide whether to add to the head or the tail of the path
            if head_dist <= tail_dist:
                # Add to the head
                path.insert(0, head_neighbor)
                head = head_neighbor
                visited.add(head_neighbor)
            else:
                # Add to the tail
                path.append(tail_neighbor)
                tail = tail_neighbor
                visited.add(tail_neighbor)
                
        return path