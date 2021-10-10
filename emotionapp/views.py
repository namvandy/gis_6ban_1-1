from django.contrib.auth.decorators import login_required
from django.shortcuts import render

# Create your views here.
from django.urls import reverse_lazy, reverse
from django.utils.decorators import method_decorator
from django.views.generic import CreateView, DetailView, ListView, UpdateView
from django.views.generic.edit import FormMixin, DeleteView
from django.views.generic.list import MultipleObjectMixin

from articleapp.models import Article
from emotionapp.decorators import emotion_ownership_required
from emotionapp.forms import EmotionCreationForm
from emotionapp.models import Emotion
from projectapp.models import Project
from subscribeapp.models import Subscription


@method_decorator(login_required, 'get')
@method_decorator(login_required, 'post')
class EmotionCreateView(CreateView):
    model = Emotion
    form_class = EmotionCreationForm
    # success_url = reverse_lazy('articleapp:list') login_required를 적용하였으므로 아래의와 같이 get_success_url 메소드를 재정의한다.
    template_name = 'emotionapp/create.html'
    def form_valid(self, form):
        form.instance.writer = self.request.user #request를 보내는 user로 writer를 할당
        return super().form_valid(form)
    def get_success_url(self):
        return reverse("emotionapp:detail" , kwargs={"pk":self.object.pk})


class EmotionListView(ListView):
    model = Emotion
    context_object_name = 'emotion_list'
    template_name = 'emotionapp/list.html'
    paginate_by = 20

@method_decorator(emotion_ownership_required,'get')
@method_decorator(emotion_ownership_required,'post')
class EmotionUpdateView(UpdateView):
    model = Emotion
    form_class = EmotionCreationForm
    context_object_name = 'target_emotion'
    template_name = 'emotionapp/update.html'
    def get_success_url(self):
        return reverse('emotionapp/detail_article.html', kwargs={'pk':self.object.pk})

@method_decorator(emotion_ownership_required,'get')
@method_decorator(emotion_ownership_required,'post')
class EmotionDeleteView(DeleteView):
    model = Emotion
    context_object_name = 'target_emotion'
    success_url = reverse_lazy('emotionapp:list')
    template_name = 'emotionapp/delete.html'

class EmotionDetailView(DetailView):
    model = Emotion
    context_object_name = 'target_emotion'
    template_name = 'emotionapp/detail.html'
    paginate_by = 20

    def get_context_data(self, **kwargs):
        emotion = self.object.emotion
        article_list = None
        if emotion == 'joy':article_list = Article.objects.filter(joy__gt=0.7) # 기쁨
        elif emotion == 'sadness':article_list = Article.objects.filter(sadness__gt=0.7) # 슬픔
        elif emotion == 'surprise':article_list = Article.objects.filter(surprise__gt=0.7) # 놀람
        elif emotion == 'love':article_list = Article.objects.filter(love__gt=0.7) # 상처
        elif emotion == 'anger':article_list = Article.objects.filter(anger__gt=0.7) # 분노
        elif emotion == 'fear':article_list = Article.objects.filter(fear__gt=0.7) # 두려움

        return super().get_context_data(object_list=article_list,object_name='article',**kwargs)
