run-dev:
	docker compose up -d
	while ! nc -z localhost 5432; do sleep 1; done
	uvicorn main:app --reload
