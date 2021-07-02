FROM pkuong/groupby:1.0
COPY . /app
EXPOSE 5000
ENTRYPOINT ["gunicorn"]
CMD ["-b", ":5000", "wsgi:app"]
