# base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# create user and set bash as default shell
RUN useradd -ms /bin/bash usuari

# Switch to the user to run the application.
USER usuari

# Set the working directory in the optional directory.
WORKDIR /opt/usuari

# add the local bin directory to the path
ENV PATH="/home/usuari/.local/bin:${PATH}"

# install python dependencies
COPY --chown=usuari:usuari requirements.txt /opt/usuari/
RUN python3 -m pip install --user -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# Model checkpoints
COPY --chown=usuari:usuari checkpoints/ /opt/usuari/checkpoints/
COPY --chown=usuari:usuari model_weights/ /opt/usuari/model_weights/
# Source code
COPY --chown=usuari:usuari SAMed/segment_anything/ /opt/usuari/segment_anything/
COPY --chown=usuari:usuari SAMed/sam_lora_image_encoder.py /opt/usuari/
COPY --chown=usuari:usuari SAMed/sam_lora_image_encoder_mask_decoder.py /opt/usuari/
COPY --chown=usuari:usuari experiments/docker/. /opt/usuari/scripts

# Set the default command to run when starting the container
ENTRYPOINT python3 -m process $0 $@

LABEL nl.diagnijmegen.rse.usuari.name=seg_algorithm