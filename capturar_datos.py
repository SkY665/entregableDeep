import cv2 
import os 
import imutils 
from camera import getcamera 

print('Escribe tu nombre')
personName = input()
dataPath = './data'
personPath = dataPath + '/' + personName

if os.path.exists(personPath):
    pritn('Persona ya registrar, sobreescribiendo datos...')
else: 
    os.makedirs(personPath)
    print('Nueva persona, capturando datos...')

#abrir camara
camera = getcamera()
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

#cargar el detector de rostros 
faceClassif = cv2.CascadeClassifier("rostros.xml")

while True:
    #tomar fotografias 
    ret, frame = cap.read()

    if not ret: 
        break 

    #cambiando el tamaño de la foto
    frame = imutils.resize(frame, width=640)

    gray = cv2.cvtColot(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCllassif.detectMultiScale(gray, 
                                          scaleFactor=1.1,
                                          minNeightbors=5,
                                          minSize=(120, 120),
                                          maxSize=(1000, 1000))

    #dibujar un rectangulo en el rostro 
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #copia imagen 
        auxFrame = frame.copy()
        # obtener el cuadrado del rostro 
        rosto = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC )
        #guardar el rostro como imagen 
        cv2.imwrite(personPath + 'rostro_{}.jpg', format(contador), rostro)
        print('rostro_{}.jpg'.format(contador)+ 'guardado')

        contador = contador + 1
    cv2.imshow('Mi camara '. frame) 

    if conntador >= 300 or cv2.waitKey(1) &  0xFF == ord('q'):
        break
#cirra camara 
cv2.destroyAllWindows()
cap.release()

