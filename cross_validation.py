import tensorflow as tf
import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import cv2
import res_inference as infer
import Data_Loader as ld
import Label_Encoder as le
import Data_Augmentation as dt_aug
import MinMax_Scaler as mms
import time
import gc 
# Parameters



DATA_NAME = "peta_o_1"
DATA_PATH = "../Data/"+DATA_NAME
MODEL_SAVE_PATH = "./"+DATA_NAME+"_Cross_Model/"
MODEL_NAME = "ResNet50_model"
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.985
BATCH_SIZE = 32
AUGMENTATION_RATE = 0.6
MIN_KM = 1
# Cross setting
CROSS_STEP = 10

EPOCH = 100
display_step = 1
save_step = 20
# Train setting
TRANSFER = False
TRAIN = True

# Network Parameters
GENDER_CLASS_NUM = 2
AGE_CLASS_NUM = 3
n_input = 50176


# image setting
CROP_HEIGHT= 224
CROP_WIDTH = 224
NUM_CHANNELS = 3

def print_confusion_matrix(confusion_matrix):
    # get confusion matrix
    # temp_sc -> sum of correct
    # temp_s -> sum
    # temp_dp -> denominator of precision
    # temp_dr -> denominator of recall
    # temp_p -> precision
    # temp_r -> recall

    class_num = confusion_matrix.shape[0]
    precision = np.zeros([class_num], dtype=float)
    recall = np.zeros([class_num], dtype=float)
    accuracy = 0
    temp_sc = 0
    temp_s = 0
    for cfm_g_r in range(0,class_num) :
        temp_sc = temp_sc +  confusion_matrix[cfm_g_r, cfm_g_r]
        temp_dp = 0
        temp_dr = 0
        for cfm_g_c in range(0,class_num) :                
            temp_dp = temp_dp + confusion_matrix[cfm_g_r, cfm_g_c]
            temp_dr = temp_dr + confusion_matrix[cfm_g_c, cfm_g_r]
        temp_s = temp_s + temp_dp
        if(temp_dp == 0):
            temp_p = 0
        else :
            temp_p = confusion_matrix[cfm_g_r, cfm_g_r]/temp_dp
        if(temp_dr == 0):
            temp_r = 0
        else :
            temp_r = confusion_matrix[cfm_g_r, cfm_g_r]/temp_dr
        
        precision[cfm_g_r] = temp_p
        recall[cfm_g_r] = temp_r
    accuracy = temp_sc/temp_s

    print("Confusion Matrix : ")
    print("Matrix\t:\t", end = '')
    for ci in range(0,class_num) :
        print("e%d\t" % (ci+1), end = '')
    print("Precision")
    for ri in range(0,class_num) :
        print("p%d\t:\t" % (ri+1), end = '')
        for ci in range(0,class_num) :
            print("%-6d\t" % (confusion_matrix[ri,ci]), end = '')
        print("%.2f%%" % (precision[ri]*100))
    print("Recall\t:\t", end = '')
    for ci in range(0,class_num) :
        print("%.2f%%\t" % (recall[ci]*100), end = '')
    print("%.2f%%" % (accuracy*100))
        
    return

def train(train_images, train_labels_gender, train_labels_age, 
            test_images, test_labels_gender, test_labels_age, save_path):

    train_acc= []
    train_loss = []
    validate_acc= []
    validate_loss = []
    shuffle=True
   # batch num
    batch_num = int(train_images.shape[0]/BATCH_SIZE)
    
    multi_task_graph=tf.Graph()
    with multi_task_graph.as_default():

        # define the placeholder
        training = tf.placeholder(tf.bool)
        image_ = tf.placeholder(tf.float32, [None, n_input],name='image-input')
        image = tf.reshape(image_, shape=[-1, CROP_HEIGHT, CROP_WIDTH, NUM_CHANNELS])

        label_gender_encode = tf.placeholder(tf.int32, [None, GENDER_CLASS_NUM], name='label_gender-input')
        label_age_encode = tf.placeholder(tf.int32, [None, AGE_CLASS_NUM], name='label_age-input')
        #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        
        # Construct model
        pred_gender_encode, pred_age_encode = infer.inference(image,  training)

        # Define global step
        global_step = tf.Variable(0, trainable=False)

        # Evaluate model
        pred_gender = tf.argmax(pred_gender_encode,1)
        pred_age = tf.argmax(pred_age_encode,1)
        expect_gender = tf.argmax(label_gender_encode,1)
        expect_age = tf.argmax(label_age_encode,1)

        correct_gender = tf.equal(pred_gender, expect_gender)
        correct_age = tf.equal(pred_age, expect_age)
        accuracy_gender = tf.reduce_mean(tf.cast(correct_gender, tf.float32))
        accuracy_age = tf.reduce_mean(tf.cast(correct_age, tf.float32))

        #Define learning rate
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, batch_num, LEARNING_RATE_DECAY, staircase=True)

        # Define loss and optimizer
        #cost_gender = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_gender_encode, labels=label_gender_encode))
        #cost_age = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_age_encode, labels=label_age_encode))

        cross_entropy_gender = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_gender_encode, labels=label_gender_encode)
        cross_entropy_age = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_age_encode, labels=label_age_encode)

        
        #count = tf.constant(0)
        
        # loss function
        def loss_function(cross_entropy, expected_labels, class_num):
            # size -> input batch size
            # MK -> each class's num
            # NK -> 1/MK
            # NKS -> sum of NK
            # BK -> weight of each class's loss
            # BK_L -> BK Limited by BK_MAX
            # BK_W -> BK weight list for loss            
            size = tf.shape(expected_labels)[0]
            MK = tf.bincount(tf.cast(expected_labels, tf.int32), minlength = class_num)
            MK_masked = tf.less(MK,MIN_KM)
            MK_L = tf.where(MK_masked, tf.fill([class_num],MIN_KM), MK)
            NK = tf.divide(1,tf.cast(MK_L, dtype=tf.float32))
            NKS = tf.reduce_sum(NK)
            BK = tf.divide(NK,NKS)
            
            BK_W = tf.zeros(size, dtype=tf.float32, name=None)
            count = 0
            def BK_set(count, expected_labels, class_num, BK, BK_W): 
                size = tf.shape(expected_labels)[0]
                zeros = tf.zeros(size, dtype=tf.float32, name="zeros")  
                masked = tf.equal(expected_labels, tf.cast(count,dtype=tf.int64))                
                temp_BK = tf.where(masked,  tf.fill([size], BK[count]), zeros)
                BK_W = tf.add(BK_W, temp_BK)
                count = count + 1
                return count, expected_labels, class_num, BK, BK_W                
            count, expected_labels, class_num, BK, BK_W = tf.while_loop((lambda count, expected_labels, class_num, BK_L, BK_W: tf.less(count, class_num)), BK_set, [count, expected_labels, class_num, BK, BK_W])

            weighted_cross_entropy = tf.multiply(cross_entropy, BK_W)
            cost = tf.reduce_mean(weighted_cross_entropy)            
            return cost
        
        # get cost
        cost_gender = loss_function(cross_entropy_gender, expect_gender, GENDER_CLASS_NUM)
        cost_age = loss_function(cross_entropy_age, expect_age, AGE_CLASS_NUM)       
        
        # choose cost age or cost gender by turns
        #cost = tf.cond(tf.equal(global_step%2,0), lambda: cost_age , lambda: cost_gender)
        # average cost
        cost = (cost_age + cost_gender)/2

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #child_optimizer = tf.contrib.layers.optimize_loss(child_cost, child_global_step, child_learning_rate, optimizer='Adam')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        
        saver = tf.train.Saver(var_list=tf.global_variables())
        # initialize
        init = tf.global_variables_initializer()
        # define session
        sess= tf.Session(graph=multi_task_graph)

    if TRANSFER :
        print ("Restore  Start!")
        saver.restore(sess,os.path.join(save_path, MODEL_NAME))
        print ("Restore  Finished!")
    else :
        print ("Initialize  Start!")
        sess.run(init)
        print ("Initialize  Finished!")

    step = 1
    while step  <= EPOCH:
        train_permutation=np.random.permutation(train_images.shape[0])
        for bp in range(batch_num):           
            # get train mini_batch                
            train_batch_images = train_images[train_permutation[bp*BATCH_SIZE:BATCH_SIZE+bp*BATCH_SIZE]]
            train_batch_labels_gender = train_labels_gender[train_permutation[bp*BATCH_SIZE:BATCH_SIZE+bp*BATCH_SIZE]]
            train_batch_labels_age = train_labels_age[train_permutation[bp*BATCH_SIZE:BATCH_SIZE+bp*BATCH_SIZE]]
            # encode label
            train_batch_labels_gender_encode = le.encode_labels(train_batch_labels_gender, GENDER_CLASS_NUM)
            train_batch_labels_age_encode = le.encode_labels(train_batch_labels_age, AGE_CLASS_NUM)
            # data augmentation
            train_batch_images_aug = dt_aug.data_augmentation(train_batch_images,AUGMENTATION_RATE)
            # get


            # Run graph
            if TRAIN :
                train_pred_gender, train_pred_age,          \
                    train_cost_total, train_cost_gender, train_cost_age,      \
                    train_acc_gender, train_acc_age, opt =  \
                    sess.run([pred_gender, pred_age,        \
                    cost, cost_gender, cost_age,                  \
                    accuracy_gender, accuracy_age, optimizer],      \
                    feed_dict={training:True,               \
                    image: train_batch_images_aug,          \
                    label_gender_encode: train_batch_labels_gender_encode, \
                    label_age_encode: train_batch_labels_age_encode})
            else :
                train_pred_gender, train_pred_age,          \
                    train_cost_total, train_cost_gender, train_cost_age,      \
                    train_acc_gender, train_acc_age =       \
                    sess.run([pred_gender, pred_age,        \
                    cost, cost_gender, cost_age,                  \
                    accuracy_gender, accuracy_age],         \
                    feed_dict={training:True,               \
                    image: train_batch_images_aug,          \
                    label_gender_encode: train_batch_labels_gender_encode, \
                    label_age_encode: train_batch_labels_age_encode})
            #print(ttfg)
        # get Validate mini batch 
        validate_permutation = np.random.permutation(test_images.shape[0])
        validate_batch_images = test_images[validate_permutation[0:BATCH_SIZE]]
        validate_batch_labels_gender = test_labels_gender[validate_permutation[0:BATCH_SIZE]]
        validate_batch_labels_age = test_labels_age[validate_permutation[0:BATCH_SIZE]]
        # encode label
        validate_batch_labels_gender_encode = le.encode_labels(validate_batch_labels_gender, GENDER_CLASS_NUM)
        validate_batch_labels_age_encode = le.encode_labels(validate_batch_labels_age, AGE_CLASS_NUM)            
        # data augmentation
        validate_batch_images_aug = dt_aug.data_augmentation(validate_batch_images,0)
        # run graph
        now_learning_rate, validate_pred_gender, validate_pred_age,            \
            validate_cost_total, validate_cost_gender, validate_cost_age,        \
            validate_acc_gender, validate_acc_age =         \
            sess.run([learning_rate, pred_gender, pred_age, \
            cost, cost_gender, cost_age,                          \
            accuracy_gender, accuracy_age],                 \
            feed_dict={training:False,                       \
            image: validate_batch_images_aug,               \
            label_gender_encode: validate_batch_labels_gender_encode, \
            label_age_encode: validate_batch_labels_age_encode})  


        if step % display_step == 0:
            #print(temp_acc_avg.shape)
            #print(temp_loss_avg.shape)
            print("Epoch: " + str(step) +" Iter " + str(step*batch_num) +\
                "\nTraining Total Loss \t= {:.12f}".format(train_cost_total) +\
                "\nValidating Total Loss \t= {:.12f}".format(validate_cost_total) +\
                "\nTraining Gender Loss \t= {:.12f}".format(train_cost_gender) + ", Training Gender Accuracy \t= {:.5f}".format(train_acc_gender) + \
                "\nValidating Gender Loss \t= {:.12f}".format(validate_cost_gender) + ", Validating Gender Accuracy \t= {:.5f}".format(validate_acc_gender) + \
                "\nTraining Age Loss \t= {:.12f}".format(train_cost_age) + ", Training Age Accuracy \t= {:.5f}".format(train_acc_age) + \
                "\nValidating Age Loss \t= {:.12f}".format(validate_cost_age) + ", Validating Age Accuracy \t= {:.5f}".format(validate_acc_age))

            print("Learning Rate : ", '{0:.12f}'.format(now_learning_rate))

            print("Gender Expected Value \t: ",validate_batch_labels_gender)   
            print("Gender Predict Value \t: ",validate_pred_gender)
            print("Age Expected Value \t: ",validate_batch_labels_age)   
            print("Age Predict Value \t: ",validate_pred_age)            
            #print("yyyy Value : ",y11)
            #print("Pred : ",preddd)
        if step % save_step == 0:
            print("Save model...")
            saver.save(sess, os.path.join(save_path, MODEL_NAME))
        step += 1
    saver.save(sess, os.path.join(save_path, MODEL_NAME))
    print("Optimization Finished!")        
    print("Save model...")
    # test 
    # confusion matrix
    confusion_matrix_gender = np.zeros([GENDER_CLASS_NUM,GENDER_CLASS_NUM], dtype=float)        
    confusion_matrix_age = np.zeros([AGE_CLASS_NUM,AGE_CLASS_NUM], dtype=float)        
    
    test_batch_num = int(test_images.shape[0]/BATCH_SIZE)
    print("Testing Start !! ---------------------------------")
    for ti in range(0,test_batch_num) :
        # get test mini batch 
        test_batch_images = test_images[BATCH_SIZE*ti:BATCH_SIZE*ti+BATCH_SIZE]
        test_batch_labels_gender = test_labels_gender[BATCH_SIZE*ti:BATCH_SIZE*ti+BATCH_SIZE]
        test_batch_labels_age = test_labels_age[BATCH_SIZE*ti:BATCH_SIZE*ti+BATCH_SIZE]
        # encode label
        test_batch_labels_gender_encode = le.encode_labels(test_batch_labels_gender, GENDER_CLASS_NUM)
        test_batch_labels_age_encode = le.encode_labels(test_batch_labels_age, AGE_CLASS_NUM)            
        
        # data augmentation
        test_batch_images_aug = dt_aug.data_augmentation(test_batch_images,0)
        # run graph        
       
        test_pred_gender, test_pred_age,            \
            test_cost_gender, test_cost_age,        \
            test_acc_gender, test_acc_age =         \
            sess.run([ pred_gender, pred_age, \
            cost_gender, cost_age,                          \
            accuracy_gender, accuracy_age],                 \
            feed_dict={training:False,                       \
            image: test_batch_images_aug,               \
            label_gender_encode: test_batch_labels_gender_encode, \
            label_age_encode: test_batch_labels_age_encode})        
        
        # set confusion matrix
        for ci in range(len(test_pred_gender)):
            confusion_matrix_gender[test_pred_gender[ci],test_batch_labels_gender[ci]] +=1;
        for ci in range(len(test_pred_age)):
            confusion_matrix_age[test_pred_age[ci],test_batch_labels_age[ci]] +=1;
    sess.close()
    return confusion_matrix_gender, confusion_matrix_age
def main(argv=None):
    if not os.path.isdir(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)
    #### Loading the data
    ## Loading Training Data
    #Load Child
    data_images, data_labels_gender, data_labels_age = ld.load_data(DATA_PATH)
    num_data = data_images.shape[0]
    num_labels_gender = data_labels_gender.shape[0]
    num_labels_age = data_labels_age.shape[0]
    
    #CHECK DATA
    print("Check Data : ")
    print("Data Files Num = ", num_data)
    print("Gender Files Num = ", num_labels_gender)
    #print(data_labels_gender)
    print("Age Files Num = ", num_labels_age)
    #print(data_labels_age)
    print("MinMaxScaler Start !!")
    #Min Max Scaler Fit
    mms.mms_fit(data_images,MODEL_SAVE_PATH+"/"+MODEL_NAME+"_")
    
    #Min Max Scaler Transform Data
    data_images_mms = mms.mms_trans(data_images,MODEL_SAVE_PATH+"/"+MODEL_NAME+"_")
    print("MinMaxScaler Complete !!")
    
    sum_confusion_matrix_gender = np.zeros([GENDER_CLASS_NUM,GENDER_CLASS_NUM], dtype=float)
    sum_confusion_matrix_age = np.zeros([AGE_CLASS_NUM,AGE_CLASS_NUM], dtype=float)
    CROSS_TEST_NUM = int(num_data/CROSS_STEP)
    print("Cross Validation Start!!----------------------------------------------")
    for ci in range(0,1) :
        cross_save_path = MODEL_SAVE_PATH+"/Cross_"+str(ci+1)+"/"
        if not os.path.isdir(cross_save_path):
            os.mkdir(cross_save_path)
        print("Cross Validation %d Start : ---------------------------------------" % (ci+1))        

        test_images = data_images_mms[ci*CROSS_TEST_NUM:ci*CROSS_TEST_NUM+CROSS_TEST_NUM]
        test_labels_gender = data_labels_gender[ci*CROSS_TEST_NUM:ci*CROSS_TEST_NUM+CROSS_TEST_NUM]
        test_labels_age = data_labels_age[ci*CROSS_TEST_NUM:ci*CROSS_TEST_NUM+CROSS_TEST_NUM]
        train_images = np.concatenate((data_images_mms[0:ci*CROSS_TEST_NUM],data_images_mms[(ci+1)*CROSS_TEST_NUM:]),axis = 0)
        train_labels_gender = np.concatenate((data_labels_gender[0:ci*CROSS_TEST_NUM],data_labels_gender[(ci+1)*CROSS_TEST_NUM:]),axis = 0)
        train_labels_age = np.concatenate((data_labels_age[0:ci*CROSS_TEST_NUM],data_labels_age[(ci+1)*CROSS_TEST_NUM:]),axis = 0)
        
        print("Train images = ",train_images.shape)
        print("Test images = ",test_images.shape)
        
        tStart = time.time()#time start
        test_confusion_matrix_gender, test_confusion_matrix_age= train(train_images,train_labels_gender,train_labels_age, \
            test_images, test_labels_gender, test_labels_age, cross_save_path)
        tEnd = time.time()#time end
        
        print ("It cost %f sec" % (tEnd - tStart))#cost time
        sum_confusion_matrix_gender = sum_confusion_matrix_gender + test_confusion_matrix_gender
        sum_confusion_matrix_age = sum_confusion_matrix_age + test_confusion_matrix_age
        """
        file = open(cross_save_path+"/Confusion_Matrix.txt", "w")
        file.write("\tchild\tadult\tsenior\n")
        file.write('child\t%d\t%d\t%d\n' %(test_confusion_matrix[0,0],test_confusion_matrix[0,1],test_confusion_matrix[0,2]))
        file.write('adult\t%d\t%d\t%d\n' %(test_confusion_matrix[1,0],test_confusion_matrix[1,1],test_confusion_matrix[1,2]))
        file.write('senior\t%d\t%d\t%d\n' %(test_confusion_matrix[2,0],test_confusion_matrix[2,1],test_confusion_matrix[2,2]))
        file.close()
        """
        print("Cross Validation %d Complete!! --------------------------------------" % (ci+1))

        

        print("Test Confusion Matrix of Gender")
        print_confusion_matrix(test_confusion_matrix_gender)
        print("Test Confusion Matrix of Age")
        print_confusion_matrix(test_confusion_matrix_age)
    print("Cross Validation Complete!!  --------------------------------------------")
    print(sum_confusion_matrix_gender)
    print(sum_confusion_matrix_age)
    #files_name.append(f[:-4])
    #print('Training Record : ', len(train_acc))
    #print('Validating Record : ', len(validate_acc))
  
    #file = open(MODEL_SAVE_PATH+"/loss.txt", "a")
    #file.write("%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t\n" % ((max_train_acc*100),(max_test_acc*100),(ave_train_acc*100) ,(ave_test_acc*100)))
    #file.close()
    
if __name__ == '__main__':
    main()


