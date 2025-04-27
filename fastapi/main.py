from faker import Faker
import json

fake = Faker()

def generate_fake_data():
    data = {
        'username': fake.user_name(),
        'email': fake.email()
    }
    return data

# Generate multiple entries
num_entries = 5
fake_data = [generate_fake_data() for _ in range(num_entries)]

# Output to JSON
print(json.dumps(fake_data, indent=4))
