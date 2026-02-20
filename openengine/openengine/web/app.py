import os
from flask import Flask

def create_app():
    """Flask application factory."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    )
    app.secret_key = os.environ.get('SECRET_KEY', 'openengine-dev-key-change-in-production')

    # Register blueprints
    from openengine.web.routes import main_bp
    app.register_blueprint(main_bp)

    return app


def run():
    """Entry point for running the web UI."""
    app = create_app()
    print("\n  🚀 OpenEngine Web UI is running!")
    print("  → http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    run()
