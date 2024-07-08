import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Lista de emoções
EXPRESSIONS = ('raiva', 'nojo', 'medo', 'feliz', 'triste', 'surpresa', 'neutro')
EXPRESSIONS_COLORS = {
    'raiva': (0, 0, 255),      # vermelho
    'nojo': (0, 255, 0),       # verde
    'medo': (255, 255, 0),     # ciano
    'feliz': (0, 255, 255),    # amarelo
    'triste': (255, 0, 0),     # azul
    'surpresa': (255, 0, 255), # magenta
    'neutro': (255, 255, 255)  # branco
}

def load_model(model_path, weights_path):
    """ Carrega o modelo e seus pesos. """
    with open(model_path, "r") as file:
        model = model_from_json(file.read())
    model.load_weights(weights_path)
    return model

def detect_faces(face_detector, gray_img):
    """ Detecta faces no imagem. """
    faces = face_detector.detectMultiScale(gray_img, 1.15, 9)
    return faces

def preprocess_face(roi_gray):
    """Preprocess the face image for prediction."""
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    return img_pixels

def predict_expression(model, img_pixels):
    """ Prevê a expressão da face. """
    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    predicted_expression = EXPRESSIONS[max_index]
    return predicted_expression

def draw_face_rectangle(frame, x, y, w, h, color, label):
    """ Desenha um retângulo e o texto na face. """
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=5)
    cv2.putText(frame, label, (int(x), int(y-10)), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

def main():
    model = load_model("data/model.json", "data/weights.h5")
    face_detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            continue

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = detect_faces(face_detector, gray_img)

        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y+w, x:x+h]
            img_pixels = preprocess_face(roi_gray)

            if np.sum([roi_gray]) != 0:
                predicted_expression = predict_expression(model, img_pixels)
                color = EXPRESSIONS_COLORS.get(predicted_expression, (255, 255, 255))
                draw_face_rectangle(frame, x, y, w, h, color, predicted_expression)
            else:
                cv2.putText(frame, 'Nenhum rosto encontrado', (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        resized_frame = cv2.resize(frame, (1000, 700))
        cv2.imshow('Reconhecimento de expressoes faciais', resized_frame)

        if cv2.waitKey(10) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
