kind delete cluster --name scheduler-sim

docker build -t python-scheduler:latest .
kind create cluster --name scheduler-sim --config kind-config.yaml
kind load docker-image python-scheduler:latest --name scheduler-sim
rm -r yamls
python generate_yaml.py
kubectl apply -f yamls/
kubectl apply -f scheduler-rbac.yaml
kubectl apply -f scheduler-deployment.yaml
kubectl get pods -o wide