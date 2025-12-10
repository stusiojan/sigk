import os
import csv
import random
import numpy as np
from PIL import Image
from pyrr import Matrix44
import moderngl_window
import moderngl

# Importy z Twoich plików (zakładamy, że jesteśmy w folderze src)
from phong_window import PhongWindow
from main import TaskType

# Konfiguracja zgodnie z PDF
DATASET_SIZE = 3000          # Wymóg: 3000 obrazów
OUTPUT_DIR = "../dataset"    # Folder wyjściowy (poza src)
CSV_FILENAME = "metadata.csv"

class DataGenerator(PhongWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = 0
        
        # Upewnij się, że folder istnieje
        os.makedirs(self.output_path, exist_ok=True)
        
        # Otwórz plik CSV do zapisu etykiet
        self.csv_file = open(os.path.join(self.output_path, CSV_FILENAME), 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        # Nagłówki CSV (parametry wejściowe dla sieci)
        self.writer.writerow([
            'filename', 
            'obj_x', 'obj_y', 'obj_z',       # Pozycja obiektu
            'diff_r', 'diff_g', 'diff_b',    # Kolor rozproszenia
            'shininess',                     # Połyskliwość
            'light_x', 'light_y', 'light_z'  # Pozycja światła
        ])
        
        # Przyspieszenie renderowania (bez vsync)
        self.wnd.mouse_exclusivity = True

    def on_render(self, time: float, frame_time: float):
        if self.count >= DATASET_SIZE:
            print(f"Zakończono generowanie {DATASET_SIZE} próbek.")
            self.csv_file.close()
            self.wnd.close()
            return

        self.ctx.clear(0.0, 0.0, 0.0, 1.0) # Czarne tło
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # --- 1. Losowanie parametrów (zgodnie z PDF str. 3-4) ---
        
        # Pozycja obiektu: [-20, 20]
        obj_pos = [random.uniform(-4, 4) for _ in range(3)]
        
        # Kolor rozproszenia: [0, 255] -> normalizujemy do [0.0, 1.0] dla OpenGL
        # Uwaga: Zapisujemy do CSV wartości 0-255 (zgodnie z PDF), ale do shadera idą floaty
        diff_int = [random.uniform(0, 255) for _ in range(3)] 
        mat_diffuse = [x / 255.0 for x in diff_int]
        
        # Połyskliwość: [3, 20]
        shininess = random.uniform(3, 20)
        
        # Pozycja światła: [-20, 20]
        light_pos = [random.uniform(-20, 20) for _ in range(3)]
        
        # Stała kamera (z Twojego kodu)
        camera_pos = [5.0, 5.0, 15.0]

        # --- 2. Aktualizacja macierzy i uniformów ---
        model_matrix = Matrix44.from_translation(obj_pos)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(camera_pos, 0.0, (0.0, 1.0, 0.0))
        mvp = proj * lookat * model_matrix

        self.model_view_projection.write(mvp.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(mat_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_pos, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_pos, dtype='f4').tobytes())

        # Render
        self.vao.render()

        # --- 3. Zapis wyniku ---
        filename = f"img_{self.count:04d}.png"
        
        # Odczyt z bufora ramki
        image = Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert('RGB')
        
        # Skalowanie do 128x128 (wymóg PDF)
        image = image.resize((128, 128), Image.Resampling.LANCZOS)
        image.save(os.path.join(self.output_path, filename))

        # Zapis etykiet do CSV
        row = [filename] + obj_pos + diff_int + [shininess] + light_pos
        self.writer.writerow(row)

        self.count += 1

if __name__ == '__main__':
    # Pobieramy argumenty z definicji zadania, ale podmieniamy output_path
    args = [
        "--shaders_dir_path=../resources/shaders/phong",
        "--shader_name=phong",
        "--model_name=sphere.obj",
        f"--output_path={OUTPUT_DIR}"
    ]
    # Uruchamiamy generator
    moderngl_window.run_window_config(DataGenerator, args=args)