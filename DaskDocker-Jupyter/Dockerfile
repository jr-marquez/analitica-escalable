# Let's use the image base-notebook to build our image on top of it
FROM jupyter/base-notebook

LABEL Analitica Escalable

# Let's change to root user to install java 8
USER root

# Install java 8
RUN apt-get update \
    && echo "Updated apt-get" \
    && apt-get install -y openjdk-8-jre \
    && echo "Installed openjdk 8"
    
# Install plugins
COPY requirements.txt /
RUN pip install "dask[complete]"
RUN pip install -r /requirements.txt


# Let's change to  "$NB_USER" command so the image runs as a non root user by default
USER $NB_UID

#Let's define this parameter to install jupyter lab instead of the default juyter notebook command so we don't have to use it when running the container with the option -e
ENV JUPYTER_ENABLE_LAB=yes