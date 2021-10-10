from django.contrib.auth.models import User
from django.db import models

# Create your models here.
from projectapp.models import Project


class Article(models.Model):
    writer = models.ForeignKey(User, on_delete=models.SET_NULL, related_name='article', null=True)
    # User가 탈퇴했을 경우, 작성자 미상(게시글에서) 처럼 되게 한다.
    # OneToOne과 다른 점은 다대다도 가능하다.
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='article/',null=True)
    content = models.TextField(null=True)
    created_at = models.DateField(auto_now_add=True, null=True)

    # 어떤 게시글이 연길되어 있는지 설정
    # 게시글과 프로젝트의 연결고리
    project = models.ForeignKey(Project, on_delete=models.SET_NULL, related_name='article', null=True, blank=True)

    # likeapp 과 연결된 칼럼
    like = models.IntegerField(default=0)  # 새로운 글을 썼을 때, default 0 으로 자동 저장됨.

    # emotion : sadness, anger, love, surprise, fear, joy
    sadness = models.DecimalField(max_digits=5, decimal_places=3, null=True)
    anger = models.DecimalField(max_digits=5, decimal_places=3, null=True)
    love = models.DecimalField(max_digits=5, decimal_places=3, null=True)
    surprise = models.DecimalField(max_digits=5, decimal_places=3, null=True)
    fear = models.DecimalField(max_digits=5, decimal_places=3, null=True)
    joy = models.DecimalField(max_digits=5, decimal_places=3, null=True)

    # 위도, 경도
    lat = models.FloatField(null=True)
    lon = models.FloatField(null=True)

    # 장소
    place = models.CharField(max_length=100, null=True)

    # 공개, 비공개
    is_private = models.BooleanField(default=False, null=False)

