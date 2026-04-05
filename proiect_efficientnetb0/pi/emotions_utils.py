from collections import deque, Counter
import time

class EmotionSystem:
    def __init__(self, window_size=20):
        # Coada care ține minte ultimele 10 predicții
        self.buffer = deque(maxlen=window_size)
        self.fps_start = time.time()
        self.frame_count = 0
        self.fps = 0

    def update_buffer(self, emotion):
        """Adaugă o nouă predicție în memorie"""
        self.buffer.append(emotion)

    def get_stable_emotion(self):
        """Returnează emoția dominantă din ultimele 10 cadre"""
        if len(self.buffer) < 1:
            return "Collecting..."
        
        # Găsește cea mai frecventă emoție
        # Ex: ['happy', 'happy', 'sad', 'happy'] -> 'happy'
        counts = Counter(self.buffer)
        most_common, _ = counts.most_common(1)[0]
        return most_common

    def update_fps(self):
        """Calculează FPS-ul pentru benchmark"""
        self.frame_count += 1
        if self.frame_count >= 10:
            end = time.time()
            self.fps = self.frame_count / (end - self.fps_start)
            self.frame_count = 0
            self.fps_start = time.time()
        return self.fps
