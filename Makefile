build-image:
	docker rmi manga-image-translator || true
	docker build . --tag=manga-image-translator

run-standalone-server:
	docker run --gpus all -p 5003:5003 --ipc=host --rm \
		-v $(PWD)/result:/app/result \
		--entrypoint python \
		manga-image-translator:latest \
		-m standalone --verbose --use-gpu --host=0.0.0.0 --port=5003
