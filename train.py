import tensorflow as tf
import preprocessing as pp
import cv2
import numpy as np
import labels as l
#import matplotlib.pyplot as plt

costy=[]
num_hl1=700
num_hl2=1100
num_hl3=1700
n_classes = 36

x=tf.placeholder('float',[None,400])
y=tf.placeholder('float')

hidden_layer1={'weight':tf.Variable(tf.random_normal([400,num_hl1])),'biases':tf.random_normal([num_hl1])}
hidden_layer2={'weight':tf.Variable(tf.random_normal([num_hl1,num_hl2])),'biases':tf.random_normal([num_hl2])}
hidden_layer3={'weight':tf.Variable(tf.random_normal([num_hl2,num_hl3])),'biases':tf.random_normal([num_hl3])}

output_layer={'weight':tf.Variable(tf.random_normal([num_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}


def nn_model(data):

    l1=tf.add(tf.matmul(data,hidden_layer1['weight']),hidden_layer1['biases'])
    l1=tf.nn.relu(l1)

    l2=tf.add(tf.matmul(l1,hidden_layer2['weight']),hidden_layer2['biases'])
    l2=tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer3['weight']), hidden_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output=tf.add(tf.matmul(l3,output_layer['weight']),output_layer['biases'])

    return output

saver = tf.train.Saver()

def train_nn(x):
    prediction =  nn_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)

    epoch = 10
    batch_size=500
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(epoch):
            epoch_loss = 0
            start=0
            #if ep >= 1:
             #   saver.restore(sess,'./model.ckpt')
            for i in range(1000):
                if start> 29200:
                    start = 0
                start = start+batch_size
                ex, ey = pp.load_batch(start,start+batch_size)
                if(len(ex)<1):
                    continue;
                _, c = sess.run([optimizer, cost], feed_dict={x: ex, y: ey})
                epoch_loss += c
                #if i%100 == 0:
                 #   print("sub epoch ",i)

            costy.append(epoch_loss)
            print("Epoch : ", ep, " loss ", epoch_loss)
        saver.save(sess,'./model.ckpt')
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accu = tf.reduce_mean(tf.cast(correct, 'float'))
        tx,ty=pp.load_batch(testing=1)
        print('Accuracy : ', accu.eval({x: tx, y: ty}))
        #print(prediction.eval({x:tx[0].reshape(-1,400)})) return full array
        prediction=tf.arg_max(prediction,1)
        pred=prediction.eval({x:tx[1].reshape(-1,400)}) #return index
        print(pred)


def test_nn(x,im):
    pre=nn_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,'model.ckpt')

        img = cv2.imread(im, 0)#test
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _,img = cv2.threshold(img, 110, 255, 0)
        img=cv2.resize(img,(20, 20))
        #img=cv2.GaussianBlur(img,(3,3),0)
        cv2.imshow("test", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img= np.array(img)
        x0 = img.flatten()
        x0=np.array(x0)
        x0.reshape(-1,400)
        res = sess.run(tf.arg_max(pre.eval(feed_dict={x:[x0]}),1))
        print(l.label[res[0]])


#train_nn(x)#jo bhi comment meh h rehene dena correct dataset tere pass nhi h
test_nn(x,"A.jpg")
#yeh sab file location h
#'../char74k/EnglishFnt/English/Fnt/Sample011/img011-00378.png'
#'../caps/EnglishFnt/English/Fnt/Sample011/img011-00378.png'
#'../char74k/EnglishHnd/English/Hnd/Sample001/img011-00378.png'
