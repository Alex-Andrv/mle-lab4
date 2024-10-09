import configparser
import json
import threading
import time

from kafka import KafkaConsumer
from sqlalchemy import create_engine
from logger import Logger
from value import get_vault_secrets

SHOW_LOG = True

class PredictionConsumer:
    def __init__(self, db_engine=None):
        """
        Initialize Kafka Consumer and database connection
        """
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.database_config = configparser.ConfigParser()
        self.database_config.read("database.ini")
        secrets = get_vault_secrets()

        # Database connection
        db_config = self.database_config['DATABASE']
        db_url = f"postgresql://{secrets['user']}:{secrets['password']}@{db_config['HOST']}:{db_config['PORT']}/{secrets['dbname']}"
        self.db_engine = db_engine if db_engine else create_engine(db_url)

        # Kafka Consumer setup
        self.consumer = KafkaConsumer(
            self.database_config['KAFKA']['TOPIC'],
            bootstrap_servers=self.database_config['KAFKA']['BOOTSTRAP_SERVERS'],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.log.info("Kafka Consumer initialized")

    def store_prediction(self, prediction_data):
        """
        Store the prediction result in the database
        """
        try:
            with self.db_engine.connect() as conn:
                query = f"""
                INSERT INTO predictions (species, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex)
                VALUES ('{prediction_data['prediction']}', {prediction_data['features']['Culmen Length (mm)']}, 
                        {prediction_data['features']['Culmen Depth (mm)']}, 
                        {prediction_data['features']['Flipper Length (mm)']}, 
                        {prediction_data['features']['Body Mass (g)']}, '{prediction_data['features']['Sex']}')
                """
                conn.execute(query)
                self.log.info("Prediction stored in the database")
        except Exception as e:
            self.log.error(f"Error storing prediction in the database: {e}")

    def consume_predictions(self):
        """
        Consume and process prediction messages from Kafka
        """
        self.log.info("Listening for prediction messages...")
        for message in self.consumer:
            prediction_data = message.value
            self.log.info(f"Received prediction: {prediction_data}")
            self.store_prediction(prediction_data)

def run_consumer_in_background():
    """
    Run the Kafka consumer in a background thread.
    """
    consumer = PredictionConsumer()

    # Create a background thread to run the Kafka consumer
    consumer_thread = threading.Thread(target=consumer.consume_predictions)

    # Set the thread as a daemon so it exits when the main program exits
    consumer_thread.daemon = True

    # Start the consumer thread
    consumer_thread.start()

    # Continue the main application while the consumer listens in the background
    print("Main application continues while Kafka consumer listens in the background...")

    # Example of main application work (can be replaced with actual logic)
    while True:
        try:
            # Simulating main application work
            print("Main application is running...")
            time.sleep(10)  # Sleep for 10 seconds
        except KeyboardInterrupt:
            print("Shutting down the application...")
            break

if __name__ == "__main__":
    run_consumer_in_background()
