FROM ubuntu:latest
FROM python:3.7.5

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN apt-get -y upgrade curl
RUN apt-get install -y unzip xvfb libxi6 libgconf-2-4
RUN curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add
RUN echo "deb [arch=amd64]  http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
RUN apt-get update -y

COPY . /app
WORKDIR /app
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["gunicorn"]
CMD ["-b", ":5000", "wsgi:app"]
