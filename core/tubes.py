import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field
from config import settings


@dataclass
class Tube:
    track_id: int
    class_name: str
    start_frame: int
    end_frame: int
    bboxes: List[List[int]]
    frames: List[np.ndarray]
    masks: List[np.ndarray] = None  # Binary segmentation masks, one per frame
    _spatial_bounds: tuple = None
    
    @property
    def duration(self):
        return len(self.bboxes)
    
    @property
    def center_trajectory(self):
        centers = []
        for bbox in self.bboxes:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append((cx, cy))
        return centers
    
    @property
    def spatial_bounds(self):
        """Get overall bounding box covering entire trajectory"""
        if self._spatial_bounds is None:
            if not self.bboxes:
                self._spatial_bounds = (0, 0, 0, 0)
            else:
                x_min = min(bbox[0] for bbox in self.bboxes)
                y_min = min(bbox[1] for bbox in self.bboxes)
                x_max = max(bbox[2] for bbox in self.bboxes)
                y_max = max(bbox[3] for bbox in self.bboxes)
                self._spatial_bounds = (x_min, y_min, x_max, y_max)
        return self._spatial_bounds


@dataclass
class GroupTube:
    """
    A virtual tube that wraps multiple co-located tubes into a single activity.
    The optimizer treats it as one unit; the renderer composites all sub-tubes.
    """
    group_id: int
    sub_tubes: List[Tube]
    # Unified bounding box covering all sub-tubes each frame
    bboxes: List[List[int]] = field(default_factory=list)
    frames: List[np.ndarray] = field(default_factory=list)
    masks: List = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    class_name: str = "group"
    track_id: int = -1
    _spatial_bounds: tuple = None

    def __post_init__(self):
        if self.sub_tubes:
            self.start_frame = min(t.start_frame for t in self.sub_tubes)
            self.end_frame = max(t.end_frame for t in self.sub_tubes)
            self.track_id = self.group_id
            classes = set(t.class_name for t in self.sub_tubes)
            self.class_name = f"group({','.join(sorted(classes))})"
            self._build_unified_data()

    def _build_unified_data(self):
        """Build frame-aligned data spanning from start_frame to end_frame."""
        total_frames = self.end_frame - self.start_frame + 1
        self.bboxes = []
        self.frames = []
        self.masks = [None] * total_frames

        for f_idx in range(total_frames):
            global_frame = self.start_frame + f_idx
            x1_min, y1_min = float('inf'), float('inf')
            x2_max, y2_max = 0, 0
            found_any = False
            ref_frame = None

            for tube in self.sub_tubes:
                local = global_frame - tube.start_frame
                if 0 <= local < tube.duration:
                    bbox = tube.bboxes[local]
                    x1_min = min(x1_min, bbox[0])
                    y1_min = min(y1_min, bbox[1])
                    x2_max = max(x2_max, bbox[2])
                    y2_max = max(y2_max, bbox[3])
                    found_any = True
                    if ref_frame is None:
                        ref_frame = tube.frames[local]

            if found_any:
                self.bboxes.append([int(x1_min), int(y1_min), int(x2_max), int(y2_max)])
                self.frames.append(ref_frame)
            else:
                # Gap frame — use previous or next available
                prev_bbox = self.bboxes[-1] if self.bboxes else [0, 0, 1, 1]
                prev_frame = self.frames[-1] if self.frames else np.zeros((1, 1, 3), dtype=np.uint8)
                self.bboxes.append(prev_bbox)
                self.frames.append(prev_frame)

    @property
    def duration(self):
        return len(self.bboxes)

    @property
    def center_trajectory(self):
        return [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in self.bboxes]

    @property
    def spatial_bounds(self):
        if self._spatial_bounds is None:
            if not self.bboxes:
                self._spatial_bounds = (0, 0, 0, 0)
            else:
                self._spatial_bounds = (
                    min(b[0] for b in self.bboxes),
                    min(b[1] for b in self.bboxes),
                    max(b[2] for b in self.bboxes),
                    max(b[3] for b in self.bboxes),
                )
        return self._spatial_bounds


class TubeGenerator:
    _group_counter = 1000  # Start group IDs above normal track IDs

    def __init__(self):
        self.min_duration = settings.min_object_duration
        self.max_tube_length = settings.max_tube_length
        self.min_motion_threshold = settings.min_motion_threshold
        self.max_gap_fill = getattr(settings, 'max_gap_fill', 15)
        self.group_merge_distance = getattr(settings, 'group_merge_distance', 2.0)
        self._stats = {
            'total': 0, 'stationary_removed': 0, 'capped': 0,
            'trimmed': 0, 'gaps_filled': 0, 'groups_merged': 0,
        }
        
    @property
    def filter_stats(self):
        return self._stats
        
    def generate_tubes(self, tracks_per_frame: List[List[dict]], 
                       video_frames: List[np.ndarray]) -> List[Tube]:
        track_data = {}
        
        for frame_idx, tracks in enumerate(tracks_per_frame):
            for track in tracks:
                tid = track['track_id']
                if tid not in track_data:
                    track_data[tid] = {
                        'class_name': track['class_name'],
                        'frames': [],
                        'bboxes': [],
                        'masks': [],
                        'frame_indices': []
                    }
                
                track_data[tid]['frames'].append(video_frames[frame_idx])
                track_data[tid]['bboxes'].append(track['bbox'])
                track_data[tid]['masks'].append(track.get('mask'))
                track_data[tid]['frame_indices'].append(frame_idx)
        
        tubes = []
        for tid, data in track_data.items():
            if len(data['frames']) < self.min_duration:
                continue
            
            # Fill gaps before creating the tube
            filled = self._fill_frame_gaps(data, video_frames)
            
            # Check if any masks exist for this tube
            has_masks = any(m is not None for m in filled['masks'])
            
            tube = Tube(
                track_id=tid,
                class_name=filled['class_name'],
                start_frame=filled['frame_indices'][0],
                end_frame=filled['frame_indices'][-1],
                bboxes=filled['bboxes'],
                frames=filled['frames'],
                masks=filled['masks'] if has_masks else None
            )
            tubes.append(tube)
        
        self._stats['total'] = len(tubes)
        
        # Post-processing: filter, trim, cap, then merge groups
        tubes = self._filter_stationary(tubes)
        tubes = [self._trim_trailing_stillness(t) for t in tubes]
        tubes = [t for t in tubes if t.duration >= self.min_duration]
        tubes = self._cap_tube_lengths(tubes)
        
        # Merge co-located tubes into groups
        tubes = self._merge_co_located_tubes(tubes)
        
        return tubes

    # ── Gap Filling ────────────────────────────────────────────────────────

    def _fill_frame_gaps(self, track_data: dict, video_frames: List[np.ndarray]) -> dict:
        """
        Fill small gaps in a track's frame sequence by interpolating bboxes
        and pulling the actual video frame for those missing indices.
        """
        indices = track_data['frame_indices']
        if len(indices) < 2:
            return track_data

        new_frames = []
        new_bboxes = []
        new_masks = []
        new_indices = []
        gaps_filled_here = 0

        for i in range(len(indices)):
            # Add existing data point
            new_frames.append(track_data['frames'][i])
            new_bboxes.append(track_data['bboxes'][i])
            new_masks.append(track_data['masks'][i])
            new_indices.append(indices[i])

            # Check for gap to next frame
            if i < len(indices) - 1:
                gap_size = indices[i + 1] - indices[i] - 1
                if 0 < gap_size <= self.max_gap_fill:
                    # Interpolate across the gap
                    bbox_start = track_data['bboxes'][i]
                    bbox_end = track_data['bboxes'][i + 1]

                    for g in range(1, gap_size + 1):
                        alpha = g / (gap_size + 1)
                        interp_bbox = [
                            int(bbox_start[0] + alpha * (bbox_end[0] - bbox_start[0])),
                            int(bbox_start[1] + alpha * (bbox_end[1] - bbox_start[1])),
                            int(bbox_start[2] + alpha * (bbox_end[2] - bbox_start[2])),
                            int(bbox_start[3] + alpha * (bbox_end[3] - bbox_start[3])),
                        ]
                        fill_idx = indices[i] + g
                        # Pull the actual video frame for this index
                        if fill_idx < len(video_frames):
                            new_frames.append(video_frames[fill_idx])
                        else:
                            new_frames.append(track_data['frames'][i])
                        new_bboxes.append(interp_bbox)
                        new_masks.append(None)  # No mask for interpolated frames
                        new_indices.append(fill_idx)
                        gaps_filled_here += 1

        if gaps_filled_here > 0:
            self._stats['gaps_filled'] += gaps_filled_here

        return {
            'class_name': track_data['class_name'],
            'frames': new_frames,
            'bboxes': new_bboxes,
            'masks': new_masks,
            'frame_indices': new_indices,
        }

    # ── Group Merging ──────────────────────────────────────────────────────

    def _merge_co_located_tubes(self, tubes: List[Tube]) -> list:
        """
        Merge tubes that overlap temporally AND are spatially close into GroupTubes.
        A group of people walking together becomes one unit in the synopsis.
        """
        if len(tubes) < 2:
            return tubes

        n = len(tubes)
        merged_into = list(range(n))  # Union-Find parent array

        def find(x):
            while merged_into[x] != x:
                merged_into[x] = merged_into[merged_into[x]]
                x = merged_into[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                merged_into[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(tubes[i], tubes[j]):
                    union(i, j)

        # Group tubes by their root
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        result = []
        groups_formed = 0
        for root, members in groups.items():
            if len(members) == 1:
                result.append(tubes[members[0]])
            else:
                group_id = TubeGenerator._group_counter
                TubeGenerator._group_counter += 1
                sub_tubes = [tubes[m] for m in members]
                gt = GroupTube(group_id=group_id, sub_tubes=sub_tubes)
                result.append(gt)
                groups_formed += 1

        if groups_formed > 0:
            self._stats['groups_merged'] = groups_formed
            print(f"  Merged {sum(len(g) for g in groups.values() if len(g) > 1)} tubes "
                  f"into {groups_formed} group(s)")

        return result

    def _should_merge(self, t1: Tube, t2: Tube) -> bool:
        """Check if two tubes should be merged (temporal overlap + spatial proximity)."""
        # Temporal overlap check — require ≥50% overlap
        overlap_start = max(t1.start_frame, t2.start_frame)
        overlap_end = min(t1.end_frame, t2.end_frame)
        overlap = max(0, overlap_end - overlap_start)
        
        shorter_duration = min(t1.end_frame - t1.start_frame, t2.end_frame - t2.start_frame)
        if shorter_duration <= 0 or overlap / shorter_duration < 0.5:
            return False

        # Spatial proximity check — average distance between centers
        avg_diag = (self._get_object_diagonal(t1) + self._get_object_diagonal(t2)) / 2
        threshold = avg_diag * self.group_merge_distance

        # Sample centers at overlap frames
        distances = []
        sample_step = max(1, overlap // 10)
        for f in range(overlap_start, overlap_end, sample_step):
            idx1 = f - t1.start_frame
            idx2 = f - t2.start_frame
            if 0 <= idx1 < len(t1.bboxes) and 0 <= idx2 < len(t2.bboxes):
                c1 = ((t1.bboxes[idx1][0] + t1.bboxes[idx1][2]) / 2,
                       (t1.bboxes[idx1][1] + t1.bboxes[idx1][3]) / 2)
                c2 = ((t2.bboxes[idx2][0] + t2.bboxes[idx2][2]) / 2,
                       (t2.bboxes[idx2][1] + t2.bboxes[idx2][3]) / 2)
                dist = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5
                distances.append(dist)

        if not distances:
            return False

        avg_distance = sum(distances) / len(distances)
        return avg_distance < threshold

    # ── Existing Methods (unchanged) ───────────────────────────────────────
    
    def _compute_displacement(self, tube: Tube) -> float:
        """Compute total center displacement across trajectory."""
        centers = tube.center_trajectory
        if len(centers) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            total += (dx**2 + dy**2) ** 0.5
        return total
    
    def _get_object_diagonal(self, tube) -> float:
        """Average diagonal size of the object across the tube."""
        diags = []
        for bbox in tube.bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            diags.append((w**2 + h**2) ** 0.5)
        return sum(diags) / len(diags) if diags else 1.0
    
    def _filter_stationary(self, tubes: List[Tube]) -> List[Tube]:
        """Remove tubes where the object barely moves relative to its size."""
        filtered = []
        for tube in tubes:
            displacement = self._compute_displacement(tube)
            diagonal = self._get_object_diagonal(tube)
            
            normalized_motion = displacement / (diagonal * max(1.0, len(tube.bboxes) ** 0.5))
            
            threshold = self.min_motion_threshold
            if len(tube.bboxes) < 60:
                threshold *= 0.5
            
            if normalized_motion >= threshold:
                filtered.append(tube)
            else:
                self._stats['stationary_removed'] += 1
                self._stats.setdefault('removed_details', []).append({
                    'track_id': tube.track_id,
                    'class': tube.class_name,
                    'duration': len(tube.bboxes),
                    'motion_score': round(normalized_motion, 4),
                    'threshold': round(threshold, 4),
                })
        
        return filtered
    
    def _trim_trailing_stillness(self, tube: Tube) -> Tube:
        """Trim frames from the end of a tube where the object has stopped moving."""
        centers = tube.center_trajectory
        if len(centers) < 2:
            return tube
        
        diagonal = self._get_object_diagonal(tube)
        still_threshold = diagonal * 0.02
        
        last_moving_idx = len(centers) - 1
        for i in range(len(centers) - 1, 0, -1):
            dx = abs(centers[i][0] - centers[i-1][0])
            dy = abs(centers[i][1] - centers[i-1][1])
            frame_displacement = (dx**2 + dy**2) ** 0.5
            
            if frame_displacement > still_threshold:
                last_moving_idx = i
                break
        else:
            return tube
        
        trim_end = min(last_moving_idx + 10, len(tube.bboxes))
        
        if trim_end < len(tube.bboxes):
            self._stats['trimmed'] += 1
            return Tube(
                track_id=tube.track_id,
                class_name=tube.class_name,
                start_frame=tube.start_frame,
                end_frame=tube.start_frame + trim_end - 1,
                bboxes=tube.bboxes[:trim_end],
                frames=tube.frames[:trim_end],
                masks=tube.masks[:trim_end] if tube.masks else None
            )
        
        return tube
    
    def _cap_tube_lengths(self, tubes: List[Tube]) -> List[Tube]:
        """Cap tubes to max_tube_length, keeping the highest-motion window."""
        capped = []
        for tube in tubes:
            if len(tube.bboxes) <= self.max_tube_length:
                capped.append(tube)
                continue
            
            self._stats['capped'] += 1
            
            centers = tube.center_trajectory
            best_start = 0
            best_motion = 0.0
            
            for start in range(0, len(centers) - self.max_tube_length + 1):
                window_motion = 0.0
                for i in range(start + 1, start + self.max_tube_length):
                    dx = centers[i][0] - centers[i-1][0]
                    dy = centers[i][1] - centers[i-1][1]
                    window_motion += (dx**2 + dy**2) ** 0.5
                
                if window_motion > best_motion:
                    best_motion = window_motion
                    best_start = start
            
            end = best_start + self.max_tube_length
            capped.append(Tube(
                track_id=tube.track_id,
                class_name=tube.class_name,
                start_frame=tube.start_frame + best_start,
                end_frame=tube.start_frame + end - 1,
                bboxes=tube.bboxes[best_start:end],
                frames=tube.frames[best_start:end],
                masks=tube.masks[best_start:end] if tube.masks else None
            ))
        
        return capped
    
    def extract_object_patches(self, tube: Tube) -> List[np.ndarray]:
        patches = []
        for frame, bbox in zip(tube.frames, tube.bboxes):
            x1, y1, x2, y2 = bbox
            patch = frame[y1:y2, x1:x2]
            patches.append(patch)
        return patches
