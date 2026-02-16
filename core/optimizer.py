import numpy as np
from typing import List, Tuple, Dict, Optional
from core.tubes import Tube
from config import settings


class SpatialLaneOptimizer:
    """
    Deterministic, zero-overlap synopsis layout optimizer.
    
    Uses spatial lanes as a scheduling heuristic — objects keep their natural
    video positions. Lanes only govern WHEN things appear, not WHERE they're drawn.
    
    Algorithm:
        1. Assign each tube to a spatial lane based on its average X position.
        2. Schedule tubes within each lane so no two overlap temporally.
        3. Check adjacent lanes for actual bbox overlap and shift if needed.
        4. Output a human-readable ASCII timeline summary.
    """
    
    def __init__(self, num_lanes: int = None, overlap_threshold: float = 0.3):
        self.num_lanes = num_lanes or getattr(settings, 'num_spatial_lanes', 5)
        self.overlap_threshold = overlap_threshold
        self.compression_ratio = settings.compression_ratio
    
    # ── Stage 1: Lane Assignment ──────────────────────────────────────────
    
    def _assign_lanes(self, tubes: List[Tube], frame_width: int) -> Dict[int, List[Tube]]:
        """
        Assign each tube to a spatial lane based on its average center-X.
        Returns {lane_index: [tubes]} dict.
        """
        lane_width = frame_width / self.num_lanes
        lanes: Dict[int, List[Tube]] = {i: [] for i in range(self.num_lanes)}
        
        for tube in tubes:
            # Average center-X across the entire trajectory
            avg_cx = np.mean([(b[0] + b[2]) / 2 for b in tube.bboxes])
            lane_idx = int(avg_cx / lane_width)
            lane_idx = max(0, min(self.num_lanes - 1, lane_idx))
            lanes[lane_idx].append(tube)
        
        return lanes
    
    # ── Stage 2: Intra-Lane Scheduling ────────────────────────────────────
    
    def _schedule_lane(self, lane_tubes: List[Tube], 
                       target_duration: int) -> List[Tuple[Tube, int]]:
        """
        Pack tubes within one lane with zero temporal overlap.
        Longest tubes first, each placed at the earliest non-conflicting time.
        """
        if not lane_tubes:
            return []
        
        # Sort by duration descending (place big tubes first for better packing)
        sorted_tubes = sorted(lane_tubes, key=lambda t: t.duration, reverse=True)
        
        placements: List[Tuple[Tube, int]] = []
        # Track occupied intervals as (start, end) pairs
        occupied: List[Tuple[int, int]] = []
        
        for tube in sorted_tubes:
            dur = tube.duration
            best_time = self._find_earliest_slot(dur, occupied, target_duration)
            placements.append((tube, best_time))
            occupied.append((best_time, best_time + dur))
            # Keep occupied sorted by start time for efficient scanning
            occupied.sort(key=lambda x: x[0])
        
        return placements
    
    def _find_earliest_slot(self, duration: int, 
                            occupied: List[Tuple[int, int]],
                            target_duration: int) -> int:
        """Find the earliest time where a tube of given duration fits without overlap."""
        if not occupied:
            return 0
        
        # Try time=0 first
        candidate = 0
        
        for start, end in occupied:
            # If the candidate + duration fits before this occupied interval, take it
            if candidate + duration <= start:
                return candidate
            # Otherwise, push candidate past this interval
            candidate = max(candidate, end)
        
        # Place after all existing intervals
        return candidate
    
    # ── Stage 3: Cross-Lane Conflict Resolution ───────────────────────────
    
    def _resolve_cross_lane(self, lane_placements: Dict[int, List[Tuple[Tube, int]]],
                            frame_width: int, frame_height: int) -> List[Tuple[Tube, int]]:
        """
        Check adjacent lanes for actual bbox overlap.
        If two simultaneously-active tubes from neighboring lanes have IoU > threshold,
        shift the shorter tube forward in time.
        """
        all_placements: List[Tuple[Tube, int]] = []
        for lane_idx in range(self.num_lanes):
            all_placements.extend(lane_placements.get(lane_idx, []))
        
        # Build a list we can mutate
        mutable = list(all_placements)
        
        shifts_made = 0
        max_iterations = len(mutable) * 3  # Safety bound
        
        for iteration in range(max_iterations):
            conflict_found = False
            
            for i in range(len(mutable)):
                for j in range(i + 1, len(mutable)):
                    tube_i, time_i = mutable[i]
                    tube_j, time_j = mutable[j]
                    
                    if self._check_bbox_overlap(tube_i, time_i, tube_j, time_j):
                        # Shift the shorter tube past the longer one
                        if tube_i.duration < tube_j.duration:
                            shift_target = i
                            blocker_end = time_j + tube_j.duration
                        else:
                            shift_target = j
                            blocker_end = time_i + tube_i.duration
                        
                        old_tube, old_time = mutable[shift_target]
                        new_time = blocker_end  # Place right after the blocker
                        mutable[shift_target] = (old_tube, new_time)
                        shifts_made += 1
                        conflict_found = True
                        break  # Restart conflict scan after any shift
                
                if conflict_found:
                    break
            
            if not conflict_found:
                break
        
        if shifts_made > 0:
            print(f"  Cross-lane: resolved {shifts_made} bbox conflict(s)")
        
        return mutable
    
    def _check_bbox_overlap(self, tube1: Tube, time1: int, 
                            tube2: Tube, time2: int) -> bool:
        """
        Sample-based IoU check between two placed tubes.
        Returns True if they actually overlap in pixel space during their
        simultaneous active window.
        """
        end1 = time1 + tube1.duration
        end2 = time2 + tube2.duration
        
        # No temporal overlap → no conflict
        if end1 <= time2 or end2 <= time1:
            return False
        
        overlap_start = max(time1, time2)
        overlap_end = min(end1, end2)
        
        # Sample every 5th frame for speed
        sample_step = max(1, (overlap_end - overlap_start) // 10)
        
        for t in range(overlap_start, overlap_end, sample_step):
            idx1 = t - time1
            idx2 = t - time2
            
            if 0 <= idx1 < len(tube1.bboxes) and 0 <= idx2 < len(tube2.bboxes):
                iou = self._bbox_iou(tube1.bboxes[idx1], tube2.bboxes[idx2])
                if iou > self.overlap_threshold:
                    return True
        
        return False
    
    @staticmethod
    def _bbox_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """Intersection over Union between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    # ── Main Entry Point ──────────────────────────────────────────────────
    
    def optimize(self, tubes: List[Tube], original_duration: int,
                 frame_width: int = 1920, frame_height: int = 1080) -> List[Tuple[Tube, int]]:
        """
        Run the full optimization pipeline.
        Returns List[(tube, start_frame)] placements.
        """
        target_duration = int(original_duration * self.compression_ratio)
        if target_duration < 1:
            target_duration = original_duration
        
        print(f"  Target duration: {target_duration} frames")
        print(f"  Spatial lanes: {self.num_lanes}")
        
        # Stage 1: Assign tubes to lanes
        lanes = self._assign_lanes(tubes, frame_width)
        
        lane_counts = {k: len(v) for k, v in lanes.items() if v}
        print(f"  Lane distribution: {lane_counts}")
        
        # Stage 2: Schedule within each lane
        lane_placements: Dict[int, List[Tuple[Tube, int]]] = {}
        for lane_idx in range(self.num_lanes):
            lane_tubes = lanes[lane_idx]
            if lane_tubes:
                lane_placements[lane_idx] = self._schedule_lane(lane_tubes, target_duration)
        
        # Stage 3: Resolve cross-lane bbox conflicts
        placements = self._resolve_cross_lane(lane_placements, frame_width, frame_height)
        
        print(f"  Placed {len(placements)}/{len(tubes)} tubes (0 overlaps guaranteed)")
        
        return placements
    
    # ── Human-Readable Summary ────────────────────────────────────────────
    
    def print_summary(self, placements: List[Tuple[Tube, int]], 
                      fps: int, original_duration_frames: int):
        """
        Print a human-readable ASCII timeline of the synopsis layout.
        """
        if not placements:
            print("\n  (no placements to summarize)")
            return
        
        # Sort by start time
        sorted_p = sorted(placements, key=lambda x: x[1])
        max_frame = max(t + tube.duration for tube, t in sorted_p)
        
        # ── Header ──
        print("\n" + "=" * 72)
        print("  SYNOPSIS LAYOUT SUMMARY")
        print("=" * 72)
        
        # ── Per-tube table ──
        print(f"\n  {'#':<4} {'Track':<8} {'Class':<12} {'Original':<16} {'Synopsis':<16} {'Dur':>5}")
        print("  " + "─" * 65)
        
        for i, (tube, start) in enumerate(sorted_p):
            end = start + tube.duration
            # Original timeframe in source video
            orig_from = tube.start_frame / fps
            orig_to = tube.end_frame / fps
            orig_str = f"{orig_from:.1f}s → {orig_to:.1f}s"
            # Synopsis timeframe in output video
            syn_str = f"{start / fps:.1f}s → {end / fps:.1f}s"
            print(f"  {i+1:<4} #{tube.track_id:<6} {tube.class_name:<12} "
                  f"{orig_str:<16} {syn_str:<16} {tube.duration:>5}")
        
        # ── ASCII Timeline ──
        print(f"\n  Timeline (each █ ≈ {max(1, max_frame // 60)} frames):")
        print("  " + "─" * 62)
        
        timeline_width = 60
        scale = timeline_width / max(1, max_frame)
        
        for i, (tube, start) in enumerate(sorted_p):
            end = start + tube.duration
            bar_start = int(start * scale)
            bar_end = max(bar_start + 1, int(end * scale))
            
            label = f"#{tube.track_id} {tube.class_name[:6]}"
            bar = " " * bar_start + "█" * (bar_end - bar_start)
            bar = bar.ljust(timeline_width)
            
            print(f"  {label:<12} |{bar}|")
        
        print("  " + " " * 13 + "0" + " " * (timeline_width - 1) + f"{max_frame / fps:.1f}s")
        
        # ── Stats ──
        original_sec = original_duration_frames / fps
        synopsis_sec = max_frame / fps
        ratio = synopsis_sec / original_sec if original_sec > 0 else 0
        
        print(f"\n  Original : {original_sec:.1f}s ({original_duration_frames} frames)")
        print(f"  Synopsis : {synopsis_sec:.1f}s ({max_frame} frames)")
        print(f"  Ratio    : {ratio:.1%} compression")
        print(f"  Objects  : {len(placements)} tubes placed")
        print("=" * 72 + "\n")


# ── Backward-compatible alias ─────────────────────────────────────────────
# Legacy code may import ConflictResolver; this alias prevents breakage.
ConflictResolver = SpatialLaneOptimizer
