import hashlib
import json

from textwrap import dedent
from time import time
from uuid import uuid4

from flask import Flask, jsonify, request

class Blockchain(object):

    def __init__(self):
        self.chain = []
        self.current_transactions = []

    def new_block(self, proof, previous_hash=None):
        """
        Создаем новый блок в нашем Блокчейне


        :параметр proof: <int> proof полученный после использования алгоритма «Доказательство выполнения работы»
        :параметр previous_hash: (Опциональный) <str> Хэш предыдущего Блока
        :return: <dict> New Block
        """

        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        # Сбрасываем текущий список транзакций
        self.current_transactions = []

        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount):
        """
      Создает новую транзакцию для того чтобы перейти к следующему искомому Блоку

        :параметр sender: <str> Адрес отправителя
        :параметр recipient: <str> Адрес получателя
        :параметр amount: <int> Количество
        :return: <int> Индекс Блока, в который будет добавлена данная транзакция - следующий блок
        """

        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })

        return self.last_block['index'] + 1

    def proof_of_work(self, last_proof):
        """
        Простой алгоритм Proof of Work:
         - Ищем число p' такое, чтобы hash(pp') содержал в себе 4 лидирующих нуля, где p это предыдущий p'
         - p это предыдущий proof, а p' это новый proof

        :параметр last_proof: <int>
        :return: <int>
        """

        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1

        return proof

    @staticmethod
    def valid_proof(last_proof, proof):
        """
        Проверяем Proof: Содержит ли hash(last_proof, proof) 4 лидирующих нуля?

        :параметр last_proof: <int> предыдущий Proof
        :параметр proof: <int> Тукущий Proof
        :return: <bool> True если все верно, иначе False.
        """

        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    @staticmethod
    def hash(block):
        """
        Создает a SHA-256 хэш блока

        :параметр block: <dict> Блок
        :return: <str>
        """

        # Мы должны быть уверены что наш Словарь упорядочен, или мы можем непоследовательные хэши
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]


# Создаем экземпляр нашего узла
app = Flask(__name__)

# Генерируем уникальный глобальный адрес для этого узла
node_identifier = str(uuid4()).replace('-', '')

# Создаем экземпляр Blockchain
blockchain = Blockchain()


@app.route('/mine', methods=['GET'])
def mine():
    # Мы запускаем алгоритм PoW для того чтобы найти следующий proof...
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    # Мы должны получить награду за найденный proof.
    # Если sender = "0", то это означает что данный узел заработал биткойн.
    blockchain.new_transaction(
        sender="0",
        recipient=node_identifier,
        amount=1,
    )

    # Формируем новый блок, путем добавления его в цепочку
    block = blockchain.new_block(proof)

    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200


@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    # Проверяем, что обязательные поля переданы в POST-запрос
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return 'Missing values', 400

    # Создаем новую транзакцию
    index = blockchain.new_transaction(values['sender'], values['recipient'], values['amount'])

    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201

@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)