from keras.layers import Conv1D, MaxPooling1D, Dropout, Input
from keras.models import Model
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization


def getNetwork(maxLen, numClasses, maskSize=2):                  
    
    inputs = Input(shape=(maxLen,1))
    
    conv1 = Conv1D(filters=8, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(inputs)
    pool1 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv1)
    norm1 = BatchNormalization()(pool1)

    conv2 = Conv1D(filters=16, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm1)
    pool2 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv2)
    norm2 = BatchNormalization()(pool2)

    conv3 = Conv1D(filters=32, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm2)
    pool3 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv3)
    norm3 = BatchNormalization()(pool3)

    conv4 = Conv1D(filters=64, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm3)
    pool4 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv4)
    norm4 = BatchNormalization()(pool4)

    conv5 = Conv1D(filters=128, kernel_size=(maskSize), strides=(2), padding='valid', activation='relu')(norm4)
    pool5 = MaxPooling1D(pool_size=(2), strides=(2), padding='valid')(conv5)
    norm5 = BatchNormalization()(pool5)

    flat6 = Flatten()(norm5)
    dens6 = Dense(256, activation='relu')(flat6)
    drop6 = Dropout(0.4)(dens6)
    norm6 = BatchNormalization()(drop6)
    
    dens7 = Dense(128, activation='relu')(norm6)
    drop7 = Dropout(0.4)(dens7)
    norm7 = BatchNormalization()(drop7)

    dens8 = Dense(64, activation='relu')(norm7)
    drop8 = Dropout(0.4)(dens8)
    norm8 = BatchNormalization()(drop8)

    dens9 = Dense(numClasses, activation='softmax')(norm8)
    
    model = Model(inputs=inputs, outputs=dens9)
    
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['mse'])

    return model