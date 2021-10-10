FROM python:3.8.2

WORKDIR /home/

RUN echo "putty10"

RUN git clone https://github.com/dhkangBsn/gis_6ban_1-1.git

WORKDIR /home/gis_6ban_1-1/

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install gunicorn

RUN pip install mysqlclient

EXPOSE 8000

CMD ["bash", "-c", "python manage.py makemigrations && python manage.py migrate --settings=gis_6ban_1.settings.deploy && python manage.py collectstatic --noinput --settings=gis_6ban_1.settings.deploy && gunicorn --env DJANGO_SETTINGS_MODULE=gis_6ban_1.settings.deploy gis_6ban_1.wsgi --bind 0.0.0.0:8000"]