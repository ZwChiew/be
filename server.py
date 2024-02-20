from flask import Flask, request, jsonify
from flask_cors import CORS
from deploy import update, main


app = Flask(__name__)
CORS(app)

@app.route('/api/noti', methods=['POST'])
def update_KB():
    try:
        update()
        print("Backend Updated")
        return "Backend Updated"
    except Exception as e:
        return jsonify({'error': e}), 500


@app.route('/api/query', methods=['POST'])
def get_query_from_react():
    try:
        data = request.get_json()
        user_input = data['data']
        print(user_input)
        err, output = main(user_input)

        if not err:
            # Return an error response to the frontend
            return jsonify({'error': output + " at backend"}), 500

        return output

    except Exception as e:
        return jsonify({'error': e + "at backend"}), 500


if __name__ == "__main__":
    app.run(debug=False)
