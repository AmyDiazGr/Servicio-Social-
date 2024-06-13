import cv2
import numpy as np
import time

class CVExample:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Inicializa la cámara

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detección de colores y etiquetado
            role_detected = self.detect_and_label(frame, hsv_image)

            # Mostrar primero el frame original
            cv2.imshow('Color Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Aplicar el protocolo si se detecta un rol
            if role_detected:
                self.apply_protocol(role_detected)

    def detect_and_label(self, frame, hsv_image):
        # Detección y etiquetado de colores para PCA, PRC y PRM
        roles = {
            'PCA': [('Azul', self.create_blue_mask), ('Morado', self.create_purple_mask)],
            'PRC': [('Rojo', self.create_red_mask), ('Amarillo', self.create_yellow_mask), ('Naranja', self.create_orange_mask)],
            'PRM': [('Verde', self.create_green_mask)]
        }
        
        for role, colors in roles.items():
            for color_name, mask_function in colors:
                mask = mask_function(hsv_image)
                percentage = self.highlight_color(mask, frame, self.color_map(color_name))
                if percentage > 0.2:
                    cv2.putText(frame, role, (10, self.get_position(role)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_map(color_name), 2, cv2.LINE_AA)
                    return role  # Retorna el rol detectado para aplicar el protocolo
        return None

    def get_position(self, role):
        positions = {
            'PCA': 100,
            'PRC': 50,
            'PRM': 150
        }
        return positions[role]

    def color_map(self, color_name):
        colors = {
            'Rojo': (0, 0, 255),
            'Verde': (0, 255, 0),
            'Azul': (255, 0, 0),
            'Amarillo': (0, 255, 255),
            'Naranja': (0, 165, 255),
            'Morado': (128, 0, 128)
        }
        return colors[color_name]

    def apply_protocol(self, role):
        # Ejemplo simple de protocolo de pulsación
        if role == 'PCA':
            self.pulsate('Azul', 'rápidas', 30)
        elif role == 'PRC':
            self.pulsate('Rojo', 'rápidas', 30)
        elif role == 'PRM':
            self.pulsate('Verde', 'rápidas', 30)

    def pulsate(self, color_name, speed, duration):
        circle_frame = np.zeros((500, 500, 3), dtype=np.uint8)
        center = (circle_frame.shape[1] // 2, circle_frame.shape[0] // 2)
        radius = 100
        end_time = time.time() + duration
        interval = 0.1 if speed == 'rápidas' else 0.5 if speed == 'medias' else 1.0
        while time.time() < end_time:
            cv2.circle(circle_frame, center, radius, self.color_map(color_name), -1)
            cv2.imshow('Pulsations', circle_frame)
            if cv2.waitKey(int(interval * 1000)) & 0xFF == ord('q'):
                break
            cv2.circle(circle_frame, center, radius, (0, 0, 0), -1)
            cv2.imshow('Pulsations', circle_frame)
            if cv2.waitKey(int(interval * 1000)) & 0xFF == ord('q'):
                break

    # Mask creation methods for each color
    def create_red_mask(self, hsv_image):
        lower_red = np.array([0, 100, 100], np.uint8)
        upper_red = np.array([10, 255, 255], np.uint8)
        return cv2.inRange(hsv_image, lower_red, upper_red)

    def create_yellow_mask(self, hsv_image):
        lower_yellow = np.array([20, 100, 100], np.uint8)
        upper_yellow = np.array([30, 255, 255], np.uint8)
        return cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    def create_orange_mask(self, hsv_image):
        lower_orange = np.array([11, 100, 100], np.uint8)
        upper_orange = np.array([19, 255, 255], np.uint8)
        return cv2.inRange(hsv_image, lower_orange, upper_orange)

    def create_blue_mask(self, hsv_image):
        lower_blue = np.array([100, 150, 0], np.uint8)
        upper_blue = np.array([140, 255, 255], np.uint8)
        return cv2.inRange(hsv_image, lower_blue, upper_blue)

    def create_purple_mask(self, hsv_image):
        lower_purple = np.array([125, 50, 50], np.uint8)
        upper_purple = np.array([145, 255, 255], np.uint8)
        return cv2.inRange(hsv_image, lower_purple, upper_purple)

    def create_green_mask(self, hsv_image):
        lower_green = np.array([50, 100, 100], np.uint8)
        upper_green = np.array([70, 255, 255], np.uint8)
        return cv2.inRange(hsv_image, lower_green, upper_green)

    def highlight_color(self, mask, img, color):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        img[np.where(mask != 0)] = color
        area_percentage = np.sum(mask) / (mask.shape[0] * mask.shape[1])
        return area_percentage

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cv_example = CVExample()
    cv_example.run()
