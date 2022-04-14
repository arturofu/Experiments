import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, SimpleRNN, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from termcolor import colored

#%%

all_chars='0123456789*'
num_features = len(all_chars)
print('no of features:', num_features)
char_to_index= dict((c,i) for i,c in enumerate(all_chars))
index_to_char= dict((i,c) for i, c in enumerate(all_chars))
#%%
index_to_char

# %%

def generate_data():
    first = np.random.randint(0,100)
    second = np.random.randint(0,100)
    example = str(first)+ '*' + str(second)
    label = str(first*second)
    return example, label
generate_data()


# %%
hidden_units=256
max_time_steps=5
model = Sequential([
    SimpleRNN(hidden_units,input_shape=(None,num_features)),
    RepeatVector(max_time_steps),
    SimpleRNN(hidden_units,return_sequences=True),
    TimeDistributed(Dense(num_features,activation='softmax'))
]
)
model.compile(
   loss='categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)
model.summary()
 # %%
def vectorize_example(example,label):
    x=np.zeros((max_time_steps,num_features))
    y=np.zeros((max_time_steps,num_features))
    
    diff_x = max_time_steps - len(example)
 
    diff_y = max_time_steps - len(label)
    
    for i,c in enumerate(example):
       
        x[i+diff_x,char_to_index[c]] =1
    for i in range(diff_x):
        x[i,char_to_index['0']] = 1


    for i,c in enumerate(label):
        y[i+diff_y,char_to_index[c]] =1
    for i in range(diff_y):
        y[i,char_to_index['0']] = 1  
    return x,y
e, l = generate_data()

print(e,l)

x,y= vectorize_example(e,l)

print(x.shape,y.shape)

# %%

def devectorize_example(example):
    result = [index_to_char[np.argmax(vec)] for i,vec in enumerate(example)]
    return ''.join(result)
devectorize_example(x)

# %%
devectorize_example(y)
# %%
def create_dataset(num_examples=10000):
    x=np.zeros((num_examples,max_time_steps,num_features))
    y=np.zeros((num_examples,max_time_steps,num_features))
    for i in range(num_examples):
        e,l = generate_data()
        e_v, l_v = vectorize_example(e,l)
        x[i] = e_v
        y[i] = l_v
    return x,y
x,y = create_dataset()
print(x.shape,y.shape)
# %%
devectorize_example(x[0])
devectorize_example(y[0])
# %%

es_cb=EarlyStopping(monitor='val_loss',patience=10)
model.fit(x,y,epochs=800,batch_size=256,validation_split=0.2,
         verbose=False,callbacks=[es_cb])


# %%
x_test,y_test = create_dataset(10)
preds = model.predict(x_test)
for i,pred in enumerate(preds):
    y=devectorize_example(y_test[i])
    y_hat = devectorize_example(pred)
    col='green'
    if y!= y_hat:
        col='red'
    out='Input: '+ devectorize_example(x_test[i])+' Out: ' +y+'Pred:' +y_hat
    print(colored(out,col))
