 jupyter nbconvert --to script solutions.ipynb
 

docker build -t penguin-flask-app .
docker run -p 5001:5001 penguin-flask-app
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d \
'{
    "Culmen Length (mm)": 39.1,
    "Culmen Depth (mm)": 18.7,
    "Flipper Length (mm)": 181,
    "Body Mass (g)": 3750,
    "Sex": "MALE",
    "Island": "Torgersen"
}'


brew services start jenkins-lts

ngrok http http://localhost:8080

cat /Users/aandreev/.jenkins/secrets/initialAdminPassword


some fix# mle-lab2


docker exec -it <vault_container_id> /bin/sh

vault kv put secret/db POSTGRES_USER=penguin_user POSTGRES_PASSWORD=penguin_password POSTGRES_DB=penguins_db

# mle-lab3
# mle-lab4
