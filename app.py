from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import datetime
from pymongo.errors import DuplicateKeyError
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import logging
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from base64 import b64encode, b64decode

from run import predict

SECRET_KEY = 'thisisasecretkey'  # 16 characters long (128-bit key)
BLOCK_SIZE = 16

app = Flask(__name__)

def encrypt_message(text, key):
    cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
    encrypted_text = cipher.encrypt(pad(text.encode('utf-8'), BLOCK_SIZE))
    return b64encode(encrypted_text).decode('utf-8')


def decrypt_message(cipher_text, key):
    cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
    decoded_encrypted_text = b64decode(cipher_text)
    decrypted_text = unpad(cipher.decrypt(decoded_encrypted_text), BLOCK_SIZE)
    return decrypted_text.decode('utf-8')


logging.basicConfig(level=logging.INFO)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

app.config[
    "MONGO_URI"] = "mongodb+srv://halil10hatun:IobZyvpsBcQRbNlS@mycluster.jzt34j1.mongodb.net/kuaikaAI?retryWrites=true&w=majority&appName=MyCluster"
mongo = PyMongo(app)

app.config["JWT_SECRET_KEY"] = "your_jwt_secret_key"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = False
jwt = JWTManager(app)

bcrypt = Bcrypt(app)

clients = mongo.db.clients
sessions = mongo.db.sessions
responses = mongo.db.responses

clients.create_index("username", unique=True)


@app.route('/')
def index():
    encrypted = encrypt_message("Hello, world!", SECRET_KEY)
    decrypted = decrypt_message("ls32WehQLhrbd29JHasmFO26sqTSv7RKFGAp4XmGvVU=", SECRET_KEY)
    text = "Welcome to the ChatBot API!" + decrypted + encrypted
    return decrypted


@app.route('/register', methods=['POST'])
def register():
    data = request.json
    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_client = {
        'name': data['name'],
        'password': hashed_password,
        'username': data['username'],
        'create_at': datetime.datetime.utcnow(),
        'update_at': datetime.datetime.utcnow()
    }
    try:
        clients.insert_one(new_client)
        return jsonify(status=True, message="Client registered successfully"), 201
    except DuplicateKeyError:
        return jsonify(status=False, message="Username already exists"), 400


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    client = clients.find_one({'username': data['username']})

    if client and bcrypt.check_password_hash(client['password'], data['password']):
        access_token = create_access_token(identity=str(client['_id']))
        return jsonify(status=True, message="Login successful", access_token=access_token,
                       client_id=str(client['_id'])), 200
    return jsonify(status=False, message="Invalid username or password"), 401


@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    client = clients.find_one({'_id': ObjectId(current_user_id)})
    if client:
        client['_id'] = str(client['_id'])
        return jsonify(status=True, message="Protected route accessed", data=client), 200
    return jsonify(status=False, message="Client not found"), 404


@app.route('/clients', methods=['GET'])
@jwt_required()
def get_clients():
    client_list = []
    for client in clients.find():
        client['_id'] = str(client['_id'])
        client_list.append(client)
    return jsonify(status=True, message="Clients retrieved successfully", data=client_list), 200


@app.route('/client/<id>', methods=['PUT'])
@jwt_required()
def update_client(id):
    data = request.json
    updated_client = {
        'name': data['name'],
        'password': bcrypt.generate_password_hash(data['password']).decode('utf-8'),
        'username': data['username'],
        'update_at': datetime.datetime.utcnow()
    }
    try:
        result = clients.update_one({'_id': ObjectId(id)}, {'$set': updated_client})
        if result.matched_count == 0:
            return jsonify(status=False, message="Client not found"), 404

        updated_client['_id'] = id
        return jsonify(status=True, message="Client updated successfully", data=updated_client), 200
    except DuplicateKeyError:
        return jsonify(status=False, message="Username already exists"), 400


@app.route('/client/<id>', methods=['DELETE'])
@jwt_required()
def delete_client(id):
    clients.delete_one({'_id': ObjectId(id)})
    return jsonify(status=True, message="Client deleted successfully"), 200


@app.route('/session', methods=['POST'])
@jwt_required()
def create_session():
    data = request.json
    new_session = {
        'Client_ID': ObjectId(data['Client_ID']),
        'Title': "New Session",
        'Start_Timestamp': datetime.datetime.utcnow(),
        'End_Timestamp': datetime.datetime.utcnow(),
        'Device': data['Device'],
        'Create_At': datetime.datetime.utcnow(),
        'Update_At': datetime.datetime.utcnow()
    }
    session_id = sessions.insert_one(new_session).inserted_id
    new_session['_id'] = str(session_id)
    new_session['Client_ID'] = str(new_session['Client_ID'])
    return jsonify(status=True, message="Session created successfully", data=new_session), 201


@app.route('/sessions', methods=['GET'])
@jwt_required()
def get_sessions():
    session_list = []
    for session in sessions.find():
        session['_id'] = str(session['_id'])
        session['Client_ID'] = str(session['Client_ID'])
        session_list.append(session)
    return jsonify(status=True, message="Sessions retrieved successfully", data=session_list), 200


@app.route('/session/<id>', methods=['PUT'])
@jwt_required()
def update_session(id):
    data = request.json
    updated_session = {
        'End_Timestamp': data.get('End_Timestamp'),
        'Device': data['Device'],
        'Update_At': datetime.datetime.utcnow()
    }
    result = sessions.update_one({'_id': ObjectId(id)}, {'$set': updated_session})

    if result.matched_count == 0:
        return jsonify(status=False, message="Session not found"), 404

    updated_session['_id'] = id
    updated_session['Client_ID'] = str(sessions.find_one({'_id': ObjectId(id)})['Client_ID'])
    return jsonify(status=True, message="Session updated successfully", data=updated_session), 200


@app.route('/session/<id>', methods=['DELETE'])
@jwt_required()
def delete_session(id):
    sessions.delete_one({'_id': ObjectId(id)})
    return jsonify(status=True, message="Session deleted successfully"), 200


@app.route('/sessions/client/<client_id>', methods=['GET'])
@jwt_required()
def get_sessions_by_client(client_id):
    try:
        client_id_obj = ObjectId(client_id)  # Client ID'yi ObjectId'ye çeviriyoruz
        session_list = []
        for session in sessions.find({'Client_ID': client_id_obj}):
            session['_id'] = str(session['_id'])
            session['Client_ID'] = str(session['Client_ID'])
            session_list.append(session)
        return jsonify(status=True, message="Sessions retrieved successfully", data=session_list), 200
    except Exception as e:
        return jsonify(status=False, message=str(e)), 422


@app.route('/response', methods=['POST'])
@jwt_required()
def create_response():
    data = request.json

    decrypted_text = decrypt_message(data['Request'], SECRET_KEY)
    response_text = predict(text=decrypted_text)
    encrypted_text = encrypt_message(response_text, SECRET_KEY)


    new_response = {
        'Session_ID': ObjectId(data['Session_ID']),
        'Request': data['Request'],
        'Response': encrypted_text,
        'Road': "", # add data descriptive
        'Timestamp': datetime.datetime.utcnow(),
        'Create_At': datetime.datetime.utcnow(),
        'Update_At': datetime.datetime.utcnow()
    }
    response_id = responses.insert_one(new_response).inserted_id
    new_response['_id'] = str(response_id)
    new_response['Session_ID'] = str(new_response['Session_ID'])
    app.logger.info("neredeyim")
    if data.get('First'):
        title_words = decrypted_text.split()[:3]
        title = ' '.join(title_words)
        app.logger.info(f'Title: {title}')
        sessions.update_one(
            {'_id': ObjectId(data['Session_ID'])},
            {'$set': {'Title': title, 'Update_At': datetime.datetime.utcnow()}}
        )

    return jsonify(status=True, message="Response created successfully", data=new_response), 201


@app.route('/responses', methods=['GET'])
@jwt_required()
def get_responses():
    response_list = []
    for response in responses.find():
        response['_id'] = str(response['_id'])
        response['Session_ID'] = str(response['Session_ID'])
        response_list.append(response)
    return jsonify(status=True, message="Responses retrieved successfully", data=response_list), 200


@app.route('/response/<id>', methods=['PUT'])
@jwt_required()
def update_response(id):
    data = request.json
    updated_response = {
        'Request': data['Request'],
        'Response': data['Response'],
        'Update_At': datetime.datetime.utcnow()
    }
    responses.update_one({'_id': ObjectId(id)}, {'$set': updated_response})
    return jsonify(status=True, message="Response updated successfully", data=updated_response), 200


@app.route('/response/<id>', methods=['DELETE'])
@jwt_required()
def delete_response(id):
    responses.delete_one({'_id': ObjectId(id)})
    return jsonify(status=True, message="Response deleted successfully"), 200


@app.route('/responses/session/<session_id>', methods=['GET'])
@jwt_required()
def get_responses_by_session(session_id):
    try:
        response_list = []
        for response in responses.find({'Session_ID': ObjectId(session_id)}):
            response['_id'] = str(response['_id'])
            response['Session_ID'] = str(response['Session_ID'])
            response_list.append(response)
        return jsonify(status=True, message="Responses retrieved successfully", data=response_list), 200
    except Exception as e:
        return jsonify(status=False, message=str(e)), 422


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)