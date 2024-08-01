from flask import render_template
from flaskr import create_app
from flaskr import db
from flask_migrate import Migrate
from flask_seeder import FlaskSeeder

app = create_app()
migrate = Migrate(app, db)
seeder = FlaskSeeder()
seeder.init_app(app, db)


if __name__ == '__main__':
    app.run(debug=True, port=8080)
