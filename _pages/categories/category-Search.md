---
title: Search
layout: archive
permalink: categories/search
author_profile: true
sidebar_main: true
---



{% assign posts = site.categories.Search %}

{% for post in posts %} {% include archive-single.html type=page.entries_layout %}{% endfor %}