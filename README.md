# Machine Learning Zoomcamp

## Deployment

- Activate virtual environment using pipenv and specify the version of python you want to use:

    `pipenv install scikit-learn==1.5.2 --python 3.11`

- To remove the virtual environment use:

    `pipenv --rm`

- Building docker image:

    `docker build -t zoomcamp-test .`

- Running docker image:

    `docker run -it --rm -p 9696:9696 zoomcamp-test`
