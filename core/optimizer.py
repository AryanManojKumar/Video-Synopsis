import numpy as np
from typing import List, Tuple, Set
from core.tubes import Tube
from config import settings
import random

class ConflictResolver:
    def __init__(self, grid_size: Tuple[int, int] = (3, 3), overlap_threshold: float = 0.3):
        self.compression_ratio = settings.compression_ratio
        self.grid_rows, self.grid_cols = grid_size
        self.overlap_threshold = overlap_threshold
        self._zone_cache = {}
        
    def _get_spatial_zones(self, bbox: List[int], frame_width: int, frame_height: int) -> Set[Tuple[int, int]]:
        """Get which grid zones this bbox occupies"""
        if frame_width <= 0 or frame_height <= 0:
            return {(0, 0)}
        
        cache_key = (tuple(bbox), frame_width, frame_height)
        if cache_key in self._zone_cache:
            return self._zone_cache[cache_key]
        
        x1, y1, x2, y2 = bbox
        
        zone_width = frame_width / self.grid_cols
        zone_height = frame_height / self.grid_rows
        
        col_start = int(x1 / zone_width)
        col_end = int(x2 / zone_width)
        row_start = int(y1 / zone_height)
        row_end = int(y2 / zone_height)
        
        zones = set()
        for row in range(max(0, row_start), min(self.grid_rows, row_end + 1)):
            for col in range(max(0, col_start), min(self.grid_cols, col_end + 1)):
                zones.add((row, col))
        
        result = zones if zones else {(0, 0)}
        self._zone_cache[cache_key] = result
        return result
    
    def _check_collision(self, tube1: Tube, time1: int, 
                        tube2: Tube, time2: int,
                        frame_width: int = 1920, frame_height: int = 1080) -> bool:
        """Check if two tubes collide in both time and space"""
        tube1_duration = len(tube1.bboxes)
        tube2_duration = len(tube2.bboxes)
        
        tube1_end = time1 + tube1_duration
        tube2_end = time2 + tube2_duration
        
        # Early exit: no temporal overlap
        if tube1_end <= time2 or tube2_end <= time1:
            return False
        
        # Find temporal overlap range
        overlap_start = max(time1, time2)
        overlap_end = min(tube1_end, tube2_end)
        
        # Sample frames in overlap (check every 5th frame for speed)
        sample_rate = 5
        for frame_time in range(overlap_start, overlap_end, sample_rate):
            idx1 = frame_time - time1
            idx2 = frame_time - time2
            
            if 0 <= idx1 < tube1_duration and 0 <= idx2 < tube2_duration:
                bbox1 = tube1.bboxes[idx1]
                bbox2 = tube2.bboxes[idx2]
                
                zones1 = self._get_spatial_zones(bbox1, frame_width, frame_height)
                zones2 = self._get_spatial_zones(bbox2, frame_width, frame_height)
                
                if zones1 & zones2:
                    overlap = self._bbox_overlap_ratio(bbox1, bbox2)
                    if overlap > self.overlap_threshold:
                        return True
        
        return False
    
    def _bbox_overlap_ratio(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU (Intersection over Union) between two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def optimize_placement(self, tubes: List[Tube], 
                          original_duration: int,
                          frame_width: int = 1920,
                          frame_height: int = 1080) -> List[Tuple[Tube, int]]:
        target_duration = int(original_duration * self.compression_ratio)
        
        if target_duration < 1:
            target_duration = original_duration
        
        tubes_sorted = sorted(tubes, key=lambda t: len(t.bboxes), reverse=True)
        
        placements = []
        
        print(f"  Target duration: {target_duration} frames")
        for idx, tube in enumerate(tubes_sorted):
            print(f"  Placing tube {idx+1}/{len(tubes_sorted)} (duration: {len(tube.bboxes)} frames)...", end='\r')
            best_time = self._find_best_placement(tube, placements, target_duration,
                                                  frame_width, frame_height)
            if best_time is not None:
                placements.append((tube, best_time))
        
        print(f"  Placed {len(placements)}/{len(tubes_sorted)} tubes" + " " * 30)
        return placements
    
    def _find_best_placement(self, tube: Tube, 
                            existing_placements: List[Tuple[Tube, int]],
                            max_duration: int,
                            frame_width: int = 1920,
                            frame_height: int = 1080) -> int:
        tube_duration = len(tube.bboxes)
        
        if tube_duration > max_duration:
            max_duration = tube_duration * 2
        
        for time in range(0, max(1, max_duration - tube_duration + 1)):
            has_collision = False
            for existing_tube, existing_time in existing_placements:
                if self._check_collision(tube, time, existing_tube, existing_time,
                                        frame_width, frame_height):
                    has_collision = True
                    break
            
            if not has_collision:
                return time
        
        return 0
    
    def optimize_genetic(self, tubes: List[Tube], 
                        original_duration: int,
                        frame_width: int = 1920,
                        frame_height: int = 1080,
                        population_size: int = 50,
                        generations: int = 100) -> List[Tuple[Tube, int]]:
        target_duration = int(original_duration * self.compression_ratio)
        
        population = self._init_population(tubes, target_duration, population_size)
        
        for gen in range(generations):
            fitness_scores = [self._fitness(ind, tubes, target_duration, 
                                           frame_width, frame_height) for ind in population]
            
            parents = self._selection(population, fitness_scores, population_size // 2)
            
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._crossover(parents[i], parents[i+1])
                    offspring.extend([child1, child2])
            
            offspring = [self._mutate(ind, target_duration, tubes) for ind in offspring]
            
            population = parents + offspring
        
        fitness_scores = [self._fitness(ind, tubes, target_duration,
                                       frame_width, frame_height) for ind in population]
        best_idx = np.argmax(fitness_scores)
        best_solution = population[best_idx]
        
        return [(tubes[i], time) for i, time in enumerate(best_solution)]
    
    def _init_population(self, tubes: List[Tube], max_duration: int, 
                        pop_size: int) -> List[List[int]]:
        population = []
        for _ in range(pop_size):
            individual = [random.randint(0, max(0, max_duration - len(tube.bboxes))) 
                         for tube in tubes]
            population.append(individual)
        return population
    
    def _fitness(self, individual: List[int], tubes: List[Tube], 
                max_duration: int, frame_width: int = 1920, 
                frame_height: int = 1080) -> float:
        collision_penalty = 0
        for i in range(len(tubes)):
            for j in range(i + 1, len(tubes)):
                if self._check_collision(tubes[i], individual[i], 
                                        tubes[j], individual[j],
                                        frame_width, frame_height):
                    collision_penalty += 1
        
        max_time = max([individual[i] + len(tubes[i].bboxes) 
                       for i in range(len(tubes))]) if tubes else 0
        duration_penalty = max(0, max_time - max_duration)
        
        compactness_bonus = max_duration / (max_time + 1)
        
        return compactness_bonus / (1.0 + collision_penalty * 10 + duration_penalty * 0.1)
    
    def _selection(self, population: List[List[int]], 
                  fitness_scores: List[float], n: int) -> List[List[int]]:
        indices = np.argsort(fitness_scores)[-n:]
        return [population[i] for i in indices]
    
    def _crossover(self, parent1: List[int], 
                  parent2: List[int]) -> Tuple[List[int], List[int]]:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def _mutate(self, individual: List[int], max_duration: int, 
               tubes: List[Tube]) -> List[int]:
        if random.random() < 0.2:
            idx = random.randint(0, len(individual) - 1)
            tube_duration = len(tubes[idx].bboxes)
            individual[idx] = random.randint(0, max(0, max_duration - tube_duration))
        return individual
