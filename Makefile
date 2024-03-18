run-dev:
	docker compose up -d
	while ! nc -z localhost 5432; do sleep 1; done
	PYTHONPATH=src alembic upgrade head
	PYTHONPATH=src uvicorn main:app --reload --log-config=src/log_conf.yaml
