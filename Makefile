run-dev:
	docker compose up -d
	sleep 5
	uvicorn main:app --reload
