# Use the Jenkins image as the base image
FROM jenkins/jenkins:lts

# Switch to root user to install dependencies
USER root

# Install prerequisites and Docker
RUN apt-get update -y && \
   apt-get install -y apt-transport-https ca-certificates curl gnupg software-properties-common && \
   curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
   echo "deb [arch=arm64] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
   apt-get update -y && \
   apt-get install -y docker-ce docker-ce-cli containerd.io && \
   apt-get clean


# # Install prerequisites
# RUN apt-get update && \
#     apt-get install -y ca-certificates curl gnupg

# # Set up Docker repository manually (no lsb_release)
# RUN install -m 0755 -d /etc/apt/keyrings && \
#     curl -fsSL https://download.docker.com/linux/debian/gpg | \
#     gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
#     chmod a+r /etc/apt/keyrings/docker.gpg && \
#     echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" \
#     > /etc/apt/sources.list.d/docker.list

# Install Docker components
RUN apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    apt-get clean

# Add Jenkins user to the Docker group (create if it doesn't exist)
RUN groupadd -f docker && \
    usermod -aG docker jenkins

# Create the Docker directory and volume for DinD
RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

# Switch back to the Jenkins user
USER jenkins