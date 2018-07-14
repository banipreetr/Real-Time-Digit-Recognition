import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Input, Flatten
import matplotlib.pyplot as plt
import tensorflow as tf
import pygame
from PIL import Image
import cv2
from resizeimage import resizeimage
import time



np.random.seed(42)

num_classes=10
input_shape = (28,28,1)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, [X_train.shape[0],X_train.shape[1],X_train.shape[2],1])
X_test = np.reshape(X_test, [X_test.shape[0],X_test.shape[1],X_test.shape[2],1])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)


def digModel(input_shape = (28, 28, 1)):
    X_input = Input(input_shape)
    X = Conv2D(filters = 32, kernel_size = (5, 5), strides = (1,1), padding = 'same',activation='relu')(X_input)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(X)

    X = Conv2D(filters = 64, kernel_size = (5,5), strides = (1,1), padding = 'same', activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(X)
    
    X = Flatten()(X)
    X = Dense(1024, activation='relu', kernel_initializer='TruncatedNormal',bias_initializer='zeros')(X)
    X = Dense(num_classes, activation='softmax', kernel_initializer='TruncatedNormal',bias_initializer='zeros')(X)
    
    model = Model(inputs = X_input, outputs = X, name='digModel')

    return model

#Normalize training and testing data
X_train = X_train/255
X_test = X_test/255

model = digModel(input_shape = (28,28,1))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#=============Uncomment the following lines if you want to train the network again!


#hist = model.fit(X_train, y_train,
#          batch_size=512,
#          epochs=10,
#          validation_data=(X_test, y_test), 
#          )
## Don't forget to set the file explorer as current directory in spyder!
#model.save_weights("weights.h5")
#==================================================================================



model.load_weights("weights.h5")



score = model.evaluate(X_test, y_test)
score2 = model.evaluate(X_train, y_train)
print("Train Accuracy: "+str(score2[1]*100)+"%")
print("Test Accuracy: "+str(score[1]*100)+"%")

#===========================End of recognition part============================

pygame.init()

display_width = 300
display_height = 300
radius = 10

black = (0,0,0) # RGB
white = (255,255,255)

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Drawing pad')

def digPredict(gameDisplay):
    #Processing the image into MNIST
    data = pygame.image.tostring(gameDisplay, 'RGBA')
    img = Image.frombytes('RGBA', (display_width, display_height), data)
    img = resizeimage.resize_cover(img, [28, 28])
    imgobj = np.asarray(img)
    imgobj = cv2.cvtColor(imgobj, cv2.COLOR_RGB2GRAY)
    (_, imgobj) = cv2.threshold(imgobj, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #Predicting
    imgobj = imgobj/255
    b = model.predict(np.reshape(imgobj,[1, imgobj.shape[0], imgobj.shape[1],1]))
    ans = np.argmax(b)
    print("Predicted Value: ",ans)
    return ans

def textObjects(text, font):
	textSurface = font.render(text, True, white)
	return textSurface, textSurface.get_rect()

def message_display(text, locx, locy,size):
	largeText = pygame.font.Font('freesansbold.ttf', size) # Font(font name, font size)
	TextSurf, TextRec = textObjects(text, largeText)
	TextRec.center = (locx,locy)
	gameDisplay.blit(TextSurf, TextRec)
	pygame.display.update()

def gameLoop():
    gameExit = False
    gameDisplay.fill(black)
    pygame.display.flip()
    tick = 0
    tock = 0
    startDraw = False
    while not gameExit:
        if tock - tick >= 2 and startDraw:
            predVal = digPredict(gameDisplay)
            gameDisplay.fill(black)
            message_display("Predicted Value: "+str(predVal), int(display_width/2), int(display_height/2), 20)
            time.sleep(2)#sleep for 5 seconds
            gameDisplay.fill(black)
            pygame.display.flip()
            tick = 0
            tock = 0
            startDraw = False
            continue
        
        tock = time.clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
        if pygame.mouse.get_pressed()[0]:
            spot = pygame.mouse.get_pos()
            pygame.draw.circle(gameDisplay,white,spot,radius)
            pygame.display.flip()
            tick = time.clock()
            startDraw = True

gameLoop()   
pygame.quit()