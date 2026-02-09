#!/bin/sh
PROJECT=llama-stack-demo
APP_NAME=eligibility
# VALUES="--values intel.yaml"
# VALUES="--values nvidia.yaml"
# VALUES=""

helm template . -f ./values-secrets.yaml --namespace ${PROJECT} --name-template ${APP_NAME} \
  --include-crds ${VALUES} 
  