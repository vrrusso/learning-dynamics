FROM pytorch/pytorch:latest

WORKDIR /src

RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6  python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN pip3 install pandas
RUN pip3 install -U scikit-learn
RUN pip3 install tqdm
RUN pip3 install imageio
RUN pip3 install matplotlib
RUN pip3 install scikit-image
RUN pip3 install --upgrade torch torchvision torchtext

COPY . .