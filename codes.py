import pandas as pd
import plotly.express as px
import streamlit as st
import sklearn
import numpy as np
import pandas as pd

st.set_page_config(layout="wide", page_icon="⚕️")

st.image("https://resources.altair.com/corp/images/industries_healthcare_header_interior_desktop.jpg", width =2300)
st.markdown(f"<h1 style='text-align:center;' >{'<b>Heart Disease Application</b>'}</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align:center;' >{'by Kim Hatem'}</h3>", unsafe_allow_html=True)

col1,col2,col3 = st.beta_columns(3)
with col1:
    select_box = st.selectbox('Navigation', ['Exploratory Analysis', 'Predictive analysis'])

url = "https://drive.google.com/file/d/1jxmdKOANm3Q-FQ4hhmX-AwkuxTiKu-XS/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)


df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

df.sex = df.sex.map({0:'female', 1:'male'})

df.chest_pain_type = df.chest_pain_type.map({1:'angina pectoris', 2:'atypical angina', 3:'non-anginal pain', 4:'SMI', 0:'absent'})

df.fasting_blood_sugar = df.fasting_blood_sugar.map({0:'lower than 120mg/ml', 1:'greater than 120mg/ml'})

df.exercise_angina = df.exercise_angina.map({0:'no', 1:'yes'})

df.st_slope = df.st_slope.map({1:'upsloping', 2:'horizontal', 3:'downsloping', 0:'absent'})

df.thalassemia = df.thalassemia.map({1:'normal', 2:'fixed defect', 3:'reversable defect', 0:'absent'})


if select_box == 'Exploratory Analysis':
    target = df.target.value_counts(normalize = True)*100
    trace1 = go.Bar(
        x = ["Has Disease", "Does not have disease"] ,
        y = target.values,
        text = target.values,
        textposition = 'auto',
        texttemplate = "%{y:.2f} %")


    fig = go.Figure(data = [trace1])
    fig.update_traces(marker=dict(color="LightBlue"))
    fig.update_layout(title_text = '<b>Target Distribution</b>',
                 xaxis_title="Target",
                yaxis_title="Percentage")
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig2 = px.histogram(df[df.target ==1], x = 'sex' , y = 'target',  color = 'sex', color_discrete_sequence =['LightBlue', 'thistle'], title="<b>Number of patients with heart disease across gender</b>")
    fig2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    df = df[df.target ==1]
    female = df[(df['sex']=='female')]
    male = df[(df['sex']=='male')]
    values = [ len(female), len(male)]
    labels = ['female', 'male']

    fig02 = px.pie(df[df.target ==1], names=labels, values=values,hole = 0.5 ,  title = "<b>% of patients with heart disease across gender</b>", color = labels, color_discrete_map={'male':'LightBlue',
                                 'female':'thistle'})


    young_patients = df[(df['age']>=29)&(df['age']<40)]
    middle_aged_patients = df[(df['age']>40)&(df['age']<55)]
    old_aged_patients = df[(df['age']>55)]


    labels = ['Young Age','Middle Aged','Old Aged']
    values = [
      len(young_patients),
      len(middle_aged_patients),
      len(old_aged_patients)
      ]

    fig03 = px.pie(df[df.target ==1], names=labels, values=values, title = "<b>% of patients with heart disease across group ages</b>", color = labels , color_discrete_map={'Young Age':'LightBlue',
                             'Middle Aged':'silver',
                             'Old Aged':'thistle'})



    st.markdown( """
    <div style ="background-color:silver;padding:0.25px">
    <h1 style ="color:black;text-align:center;">Patients Demographics</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")

    col1, col2, col3 = st.beta_columns(3)
    col1.write(fig)
    col2.write(fig02)
    col3.write(fig03)

    st.markdown( """
    <div style ="background-color:silver;padding:0.25px">
    <h1 style ="color:black;text-align:center;">Symptoms Exploration</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")


    head1, head2 = st.beta_columns(2)
    with head1:
        gender_options = ["male", "female"]
        gender_choice =  st.selectbox('Select the gender that you are interested in:', gender_options)
    with head2:
        age_options = ['Young Age','Middle Aged','Old Aged']
        age_choice = st.selectbox("Select the age group that you are interested in:", age_options)


    df = df.copy()
    df = df[df.sex== gender_choice]
    if age_choice == 'Young Age':
        df = young_patients
    if age_choice == 'Middle Aged':
        df = middle_aged_patients
    if age_choice == "Old Aged":
        df = old_aged_patients
    fig04 = px.histogram(df[df.target ==1], x = 'fasting_blood_sugar' , y = 'target',  color_discrete_sequence =['LightBlue'],  title="<b>Fasting blood sugar across patients with heart disease</b>")
    fig04.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig05 = px.histogram(df[df.target ==1], x = 'exercise_angina' , y = 'target',  color_discrete_sequence =['LightBlue'],  title="<b>Exercice induced Angina across patients with heart disease</b>")
    fig05.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig06 = px.histogram(df[df.target ==1], x = 'thalassemia' , y = 'target',  color_discrete_sequence =['LightBlue'],  title="<b>Thalium heart scan across patients with heart disease</b>")
    fig06.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})


    col4, col5, col6 = st.beta_columns(3)
    col4.write(fig04)
    col5.write(fig05)
    col6.write(fig06)


# Plotting a pie chart for age ranges of patients

    fig07 = px.histogram(df[df.target ==1], x = 'chest_pain_type' , y = 'target',  color_discrete_sequence =['LightBlue'],  title="<b>Chest pain type among patients with heart disease</b>")
    fig07.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig08 = px.histogram(df[df.target ==1], x = 'st_slope' , y = 'target',  color_discrete_sequence =['LightBlue'],  title="<b>Slope among patients with heart disease</b>")
    fig08.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})


    col7, col8, col9 = st.beta_columns(3)
    col7.write(fig07)
    col8.write(fig08)

if select_box == 'Predictive analysis':


    url = "https://drive.google.com/file/d/1jxmdKOANm3Q-FQ4hhmX-AwkuxTiKu-XS/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df = pd.read_csv(path)

    X = df.drop('target', axis=1).copy()
    y = df['target'].copy()


    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features= ['age', 'sex', 'cp', 'trestbps', 'chol', 	'fbs', 	'restecg', 	'thalach' ,	'exang', 	'oldpeak' ,	'slope', 'ca', 'thal']
    df[features] = scaler.fit_transform(df[features])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    np.random.seed(42)
    from sklearn.linear_model import LogisticRegression
    LR_clf=LogisticRegression()
    LR_clf.fit(X_train,y_train)
    LR_Y_pred=LR_clf.predict(X_test)

    def preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal ):

        # Pre-processing user input
        if sex=="male":
            sex=1
        else: sex=0

        if cp=="Typical angina":
            cp=0
        elif cp=="Atypical angina":
            cp=1
        elif cp=="Non-anginal pain":
            cp=2
        elif cp=="Asymptomatic":
            cp=2

        if exang=="Yes":
            exang=1
        elif exang=="No":
            exang=0

        if fbs=="Yes":
            fbs=1
        elif fbs=="No":
            fbs=0

        if slope=="Upsloping: better heart rate with excercise(uncommon)":
            slope=0
        elif slope=="Flatsloping: minimal change(typical healthy heart)":
              slope=1
        elif slope=="Downsloping: signs of unhealthy heart":
            slope=2

        if thal=="fixed defect: used to be defect but ok now":
            thal=6
        elif thal=="reversable defect: no proper blood movement when excercising":
            thal=7
        elif thal=="normal":
            thal=2.31

        if restecg=="Nothing to note":
            restecg=0
        elif restecg=="ST-T Wave abnormality":
            restecg=1
        elif restecg=="Possible or definite left ventricular hypertrophy":
            restecg=2

        user_input=[age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal]
        user_input=np.array(user_input)
        user_input=user_input.reshape(1,-1)
        user_input=scaler.fit_transform(user_input)
        prediction = LR_clf.predict(user_input)

        return prediction

        # front end elements of the web page
    html_temp = """
        <div style ="background-color:silver;padding:0.25px">
        <h1 style ="color:black;text-align:center;">Heart Disease Prediction</h1>
        </div>
        """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)


    # following lines create boxes in which user can enter data required to make prediction
    col1,col2,col3 = st.beta_columns(3)
    with col1:
        age=st.selectbox ("Age",range(18,100,1))
        sex = st.radio("Select Gender: ", ('male', 'female'))
        cp = st.selectbox('Chest Pain Type',("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"))
        trestbps=st.selectbox('Resting Blood Sugar',range(1,500,1))
        restecg=st.selectbox('Resting Electrocardiographic Results',("Nothing to note","ST-T Wave abnormality","Possible or definite left ventricular hypertrophy"))
    with col2:
        chol=st.selectbox('Serum Cholestoral in mg/dl',range(1,1000,1))
        fbs=st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes','No'])
        thalach=st.selectbox('Maximum Heart Rate Achieved',range(1,300,1))
        exang=st.selectbox('Exercise Induced Angina',["Yes","No"])
    with col3:
        oldpeak=st.number_input('Oldpeak')
        slope = st.selectbox('Heart Rate Slope',("Upsloping: better heart rate with excercise(uncommon)","Flatsloping: minimal change(typical healthy heart)","Downsloping: signs of unhealthy heart"))
        ca=st.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,5,1))
        thal=st.selectbox('Thalium Stress Result',range(1,8,1))

    #user_input=preprocess(sex,cp,exang, fbs, slope, thal )
    pred=preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal)

    if st.button("Predict"):
      if pred[0] == 0:
        st.error('Warning! You have high risk of getting a heart attack!')

      else:
        st.success('You have lower risk of getting a heart disease!')

    st.sidebar.subheader("About App")
    st.sidebar.info("This web app helps you find out whether you are at a risk of developing a heart disease.")
    st.sidebar.info("Enter the required fields and click on the 'Predict' button to check whether you have a healthy heart")

    st.info("Caution: This is just a prediction and not doctoral advice. Kindly consult with a doctor if you feel the symptoms persist.")
