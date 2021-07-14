# ML-Deployment
ML Deployment using Vertex AI


This project was done following the ML deployment using the Vertex AI tutorial https://codelabs.developers.google.com/codelabs/vertex-ai-custom-models#0

Below are the few things that i learnt :

**1. Introduction to setting up environment**

- Setting up Cloud console basically the Google cloud platform. 
- Create new project
- Enable billing (Please have your own credit card details ready ;) or else the a gmail id and credit card having the same user name )
- For new user enable **$300 USD Free Trial** 

- Working with Cloud shell.
- Authorize cloud shell by using:
  gcloud auth list
- Confirm that the gcloud command knows about your project by using:
  gcloud config list project
  If it is not, you can set it with this command:
  gcloud config set project <PROJECT_ID>
- Cloud Shell has a few environment variables, including GOOGLE_CLOUD_PROJECT which contains the name of our current Cloud project. You can see it by running:
  echo $GOOGLE_CLOUD_PROJECT
- Enabling APIs
  To access to the Compute Engine, Container Registry, and Vertex AI services:
  gcloud services enable compute.googleapis.com         \
                       containerregistry.googleapis.com  \
                       aiplatform.googleapis.com
- Creating cloud storage bucket
  To run a training job on Vertex AI, we'll need a storage bucket to store our saved model assets. To create bucket:
  BUCKET_NAME=gs://$GOOGLE_CLOUD_PROJECT-bucket
  gsutil mb -l us-central1 $BUCKET_NAME
- Creating alias of python3 
  To ensure you use Python 3 when you run the scripts , create an alias by :
  alias python=python3
  
**2. Write your code in a colab file.** My python file is uploaded for reference. (https://www.tensorflow.org/tutorials/keras/regression)

**3. Containerize training code**
- Setup files
  mkdir mpg
  cd mpg
  touch Dockerfile
  mkdir trainer
  touch trainer/train.py
  The file structure looks as follows
  ![image](https://user-images.githubusercontent.com/56925128/125643851-574deed8-6b01-4fba-9f60-6cc70e833009.png)

- Open Editor. To containerize our code we'll first create a Dockerfile.
  write the below in the docker file:
  FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3
  WORKDIR /

  # Copies the trainer code to the docker image.
  COPY trainer /trainer

  # Sets up the entry point to invoke the trainer.
  ENTRYPOINT ["python", "-m", "trainer.train"]
  
- Add model training code in train.py(basically from your python file)
- In the terminal in the cloud shell , run the following command to add your own bucket name to the file:
  sed -i "s|BUCKET_NAME|$BUCKET_NAME|g" trainer/train.py
- Build and test the container locally
  run the following to define a variable with the URI of your container image in Google Container Registry:
  IMAGE_URI="gcr.io/$GOOGLE_CLOUD_PROJECT/mpg:v1"
  build the container by running the following from the root of your mpg directory:
  docker build ./ -t $IMAGE_URI
  Once you've built the container, push it to Google Container Registry:
  docker push $IMAGE_URI
  ![image](https://user-images.githubusercontent.com/56925128/125645064-3e79ad89-14df-47c4-b0f6-84bcde0fe5b6.png)
  
**4. Run a training job on Vertex AI**
- Go to Vertex AI and then training
- Kick off the training job and pass all the values as per the tutorial link.
- Start training and you can training under Vertex AI - training
  ![image](https://user-images.githubusercontent.com/56925128/125645663-bcd8cc9c-25d0-4328-ba23-a41adabb8a45.png)

**5. Deploy a model endpoint**
- Install Vertex SDK by running the below command in your terminal:
  pip3 install google-cloud-aiplatform --upgrade --user
- Create model and deploy endpoint
  For this under mpg folder, create a new file deploy.py and write the following code:
  from google.cloud import aiplatform

  # Create a model resource from public model assets
  model = aiplatform.Model.upload(
      display_name="mpg-imported",
      artifact_uri="gs://io-vertex-codelab/mpg-model/",
      serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-3:latest"
  )

  # Deploy the above model to an endpoint
  endpoint = model.deploy(
      machine_type="n1-standard-4"
  )
  
- from mpg directory in the cloud shell terminal, run the below:
  python3 deploy.py | tee deploy-output.txt
  
- To ensure it's working correctly, navigate to the Models section of your console in Vertex AI
  ![image](https://user-images.githubusercontent.com/56925128/125646347-d6b9bb58-7a83-46d9-bf43-62688243585d.png)
  
  ![image](https://user-images.githubusercontent.com/56925128/125646457-79105b7a-cf6a-4d55-9ccf-e289cc5417fe.png)
  
- Get predictions on the deployed endpoint
- In your Cloud Shell editor, create a new file called predict.py and write the following code into it:
  from google.cloud import aiplatform

  endpoint = aiplatform.Endpoint(
      endpoint_name="projects/1000367416351/locations/us-central1/endpoints/5209169433053888512"
  )

  # A test example we'll send to our model for prediction
  test_mpg = [1, 2, 3, 2, -2, -1, -2, -1, 0]

  response = endpoint.predict([test_mpg])

  print('API response: ', response)

  print('Predicted MPG: ', response.predictions[0][0])
  
- In the terminal and enter the following to replace ENDPOINT_STRING in the predict file with your own endpoint:
  ENDPOINT=$(cat deploy-output.txt | sed -nre 's:.*Resource name\: (.*):\1:p' | tail -1)
  sed -i "s|ENDPOINT_STRING|$ENDPOINT|g" predict.py
  
- Run the predict.py file to get a prediction from our deployed model endpoint:
  python3 predict.py
  
- The API's response will be logged, along with the prediction.
  ![image](https://user-images.githubusercontent.com/56925128/125647162-c25b4d70-9b03-4960-a687-c304506950de.png)
  
6. Cleanup (if necessary)
- To delete the endpoint you deployed, navigate to the Endpoints section of your Vertex console and click the delete icon:
  ![image](https://user-images.githubusercontent.com/56925128/125647628-111cb78a-e7bf-4069-ba2e-37c6b677b9f5.png)
  ![image](https://user-images.githubusercontent.com/56925128/125647722-6e58a0a9-8e2e-4c7e-8be5-b4844588a86f.png)






