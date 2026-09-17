from flask import current_app, jsonify, request


def require_admin_key():
    admin_key = request.headers.get("X-Admin-Key")
    if not admin_key or admin_key != current_app.config["EDGE_ADMIN_API_KEY"]:
        return jsonify({"error": "unauthorized"}), 401
    return None
