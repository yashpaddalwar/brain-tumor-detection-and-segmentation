import streamlit as st
import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image,ImageOps

# MODEL_PATHS = ['braintumorcnnmodelchanges10.h5','braintumorMOBILENETNEWAPPROACHFINAL.h5','CNN_MODEL_0_9794.h5','braintumorDENSENET169NEWAPPROACHFINAL.h5']
# MODEL_NAMES = ['BEST_CNN','MobileNet','ResNET-152','DenseNet-169']
# MODEL_ACCURACY = [98.32,98.63,97.71,96.56]
# dec = {0:'Glioma Tumor', 1:'Meningioma Tumor', 2:'No Tumor', 3:'Pituitary Tumor'}


MODEL_PATHS = ['braintumorcnnmodelchanges10.h5','NETMOBILENEWAPPROACHFINAL.h5','braintumorDENSENET169NEWAPPROACHFINAL.h5']
MODEL_NAMES = ['Custom CNN','MobileNet','DenseNet-169']
MODEL_ACCURACY = [98.32,98.63,96.56]

MODEL_PATHS1 = ['vgg16final.h5','resnet152final.h5']
MODEL_NAMES1 = ['VGG-16','ResNET-152']
MODEL_ACCURACY1 = [98.62,97.71]
dec = {0:'Glioma Tumor', 1:'Meningioma Tumor', 2:'No Tumor', 3:'Pituitary Tumor'}

st.title('Brain Tumor Classification')

tab1,tab2,tab3 = st.tabs(['Test','Dataset Information','Model Information'])

with tab1:
    st.header("Let's test")
    im = st.file_uploader("Upload Image file!",type=['png', 'jpg'])
    if im:
        st.subheader('Your Image')
        st.image(im)
        image = Image.open(im)
        image = np.array(image)
        image = cv2.resize(image,(224,224))
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        input_arr = image/255
        st.markdown('---------------------------')
        st.subheader('Processed Image')
        st.image(input_arr)
        x = np.expand_dims(input_arr,axis=0)
        st.markdown('----------------------------')
        EACH_ACC = []
        EACH_PRED = []
        for model_path, model_name, model_acc in zip(MODEL_PATHS,MODEL_NAMES,MODEL_ACCURACY):
            model = load_model(model_path)
            # preds = model.predict(x)
            # preds = preds.reshape((4))
            # lst = list(preds)
            # acc = f"{round(max(lst)*100,2)} %"
            # temp = round(max(lst)*100,2)
            # preds = dec[lst.index(max(lst))]
            y_pred = model.predict(x).argmax(axis=1)
            prediction = model.predict(x)

            # preds = preds.reshape((4))
            # lst = list(preds)
            acc = f"{round(float(prediction.max())*100,2)} %"
            temp = round(float(prediction.max())*100,2)
            preds = dec[y_pred[0]]
            st.subheader(f'Model Name: {model_name}  |  Model Accuracy: {model_acc} %:')
            st.write(f"Output: {preds} detected with an accuracy of {acc}")
            st.markdown('--------------------------------------------')
            EACH_ACC.append(temp)
            EACH_PRED.append(preds)

        save_image_path = "./upload_images/"+im.name
        with open(save_image_path,"wb") as f:
            f.write(im.getbuffer())
        img = cv2.imread(save_image_path)
        img = cv2.resize(img,(224,224))
        img_input = img.reshape((1,224,224,3))

        for model_path, model_name, model_acc in zip(MODEL_PATHS1,MODEL_NAMES1,MODEL_ACCURACY1):
            model = load_model(model_path)
            preds = model.predict(x)
            y_pred = model.predict(img_input).argmax(axis=1)
            prediction = model.predict(img_input)

            preds = preds.reshape((4))
            lst = list(preds)
            acc = f"{round(float(prediction.max())*100,2)} %"
            temp = round(float(prediction.max())*100,2)
            preds = dec[y_pred[0]]
            st.subheader(f'Model Name: {model_name}  |  Model Accuracy: {model_acc} %:')
            st.write(f"Output: {preds} detected with an accuracy of {acc}")
            st.markdown('--------------------------------------------')
            EACH_ACC.append(temp)
            EACH_PRED.append(preds)

        # newone = cv2.resize(im,(224,224)) 
        # newone = newone.reshape((1,224,224,3))
        # m = 'CNN_MODEL_0_9794.h5'
        # m = load_model(m)
        # preds = model.predict(newone)
        # preds = preds.reshape((4))
        # lst = list(preds)
        # acc = f"{round(max(lst)*100,2)} %"
        # temp = round(max(lst)*100,2)
        # preds = dec[lst.index(max(lst))]
        # st.subheader(f'Output from ResNET-152 Architecture Model with accuracy 97.71 %:')
        # st.write(f"The Image contains {preds} with an accuracy of {acc}")

        import numpy as np
        import matplotlib.pyplot as plt
        import plotly.express as px

        st.header('Prediction Graph with accuracies')
        # creating the dataset
        data = {f'Best CNN - {EACH_PRED[0]}':EACH_ACC[0], f'MobileNET - {EACH_PRED[1]}':EACH_ACC[1], f'DenseNET - {EACH_PRED[2]}':EACH_ACC[2], f'VGG-16 - {EACH_PRED[3]}':EACH_ACC[3], f'ResNET-152 - {EACH_PRED[4]}':EACH_ACC[4],}
        Models = list(data.keys())
        Accuracies = list(data.values())
        df = pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])

        # creating the bar plot
        fig1 = px.bar(df,x='Models',y='Accuracies')
        st.plotly_chart(fig1)

with tab2:
    import streamlit as st
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.express as px

    st.header('Training Data')
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    # labels = 'Glioma Tumor (1321 Images)', 'Meningioma Tumor (1339 Images)', 'No Tumor (1595 Images)', 'Pituitary Tumor (1457 Images)'
    # sizes = [1321, 1339, 1595, 1457]
    # explode = (0, 0.1, 0, 0)

    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    # ax1.axis('equal')

    # st.pyplot(fig1)

    # st.header('Testing Data')
    # labels = 'Glioma Tumor (300 Images)', 'Meningioma Tumor (306 Images)', 'No Tumor (405 Images)', 'Pituitary Tumor (300 Images)'
    # sizes = [300, 306, 405, 300]
    # explode = (0, 0.1, 0, 0)

    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    # ax1.axis('equal')
    # st.pyplot(fig1)
    labels = ['Glioma Tumor (1321 Images)', 'Meningioma Tumor (1339 Images)', 'No Tumor (1595 Images)', 'Pituitary Tumor (1457 Images)']
    sizes = [1321, 1339, 1595, 1457]
    df = pd.DataFrame(list(zip(labels,sizes)),columns=['labels','sizes'])
    fig = px.pie(df,values='sizes',names='labels')
    st.plotly_chart(fig)
    st.markdown('----------------------------------')
    st.header('Testing Data')
    labels = 'Glioma Tumor (300 Images)', 'Meningioma Tumor (306 Images)', 'No Tumor (405 Images)', 'Pituitary Tumor (300 Images)'
    sizes = [300, 306, 405, 300]
    df = pd.DataFrame(list(zip(labels,sizes)),columns=['labels','sizes'])
    fig = px.pie(df,values='sizes',names='labels')
    st.plotly_chart(fig)


with tab3:
    import numpy as np
    import matplotlib.pyplot as plt

    st.header('Model Accuracies')
    # # data = {'Best CNN':98.35, 'MobileNET':98.03,'ResNET-152':97.71,'DenseNET-169':96.56}
    # data = {'Best CNN':98.35, 'MobileNET':98.03,'DenseNET-169':96.56}
    # courses = list(data.keys())
    # values = list(data.values())
    
    # fig1, ax1 = plt.subplots()
    # ax1.bar(courses, values, color ='maroon',
    #         width = 0.4)
    # st.pyplot(fig1)
    data = {'Best CNN':98.35, 'MobileNET':98.63,'DenseNET-169':96.56,'VGG-16':98.62,'ResNET-152':97.71}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
    fig = px.bar(df,y='Accuracies',x='Models')
    st.plotly_chart(fig)






