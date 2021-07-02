# This is needed to minimize the time it takes for EBS to build the image on the ec2 instance everytime.
image_name="groupby"
tag="2.0"

docker build -t groupby -f DockerfileBase .
docker login registry.hub.docker.com --username $DOCKER_USER --password $DOCKER_PASSWORD
docker tag $image_name registry.hub.docker.com/pkuong/$image_name:$tag
docker push registry.hub.docker.com/pkuong/$image_name:$tag
