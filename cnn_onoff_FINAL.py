import sensor,lcd,image
import KPU as kpu
import utime
from Maix import GPIO
from board import board_info
from fpioa_manager import fm

fm.register(13,fm.fpioa.GPIO3)
led_r=GPIO(GPIO.GPIO3,GPIO.OUT)

lcd.init()
sensor.reset()
sensor.set_auto_gain(0,24) # auto gain disable and 24dB
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))    #set to 224x224 input
sensor.set_vflip(True)                    #flip camera
#sensor.set_hmirror(0)               #flip camera
task = kpu.load(0x300000)           #load model from flash address 0x300000
sensor.run(1)

img = sensor.snapshot()
lcd.display(img,oft=(0,0))      #display large picture
img1=img.to_grayscale(1)        #convert to gray
img2=img1.resize(14,14)         #resize to mnist input 28x28
a=img2.invert()                 #invert picture as mnist need
a=img2.strech_char(1)           #preprocessing pictures, eliminate dark corner
lcd.display(img2,oft=(224,32))  #display small 28x28 picture
a=img2.pix_to_ai();             #generate data for ai
while True:
    led_r.value(1)
    fmap=kpu.forward(task,img2)     #run neural network model
    led_r.value(0)
    utime.sleep_ms(1)
plist=fmap[:]                   #get result (10 digit's probability)
pmax=max(plist)                 #get max probability
print(plist)
max_index=plist.index(pmax)     #get the digit
lcd.draw_string(224,0,"%d: %.3f"%(max_index,pmax),lcd.WHITE,lcd.BLACK)  #show result
