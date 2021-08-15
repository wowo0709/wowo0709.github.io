---
layout: single
title: "[Android] 5(1). 안드로이드에서 DrawView 위젯 사용하기"
---

<br>

# 안드로이드에서 DrawView 위젯 사용하기

이번 포스팅에서는 안드로이드 스튜디오에서 DrawView 위젯을 사용하는 방법에 대해 알아보겠습니다. 

DrawView 위젯은 사용자에게 손글씨를 입력받을 수 있는 위젯으로, AndroidDraw 라이브러리를 포함해야 합니다. 

<br>

#### 1. AndroidDraw 레포지토리 추가

---

라이브러리를 사용하기 위해 프로젝트의 build.gradle 파일에서 maven에 아래의 URL을 추가합니다. 

아래의 URL을 추가하면 jitpack에서 AndroidDraw 라이브러리를 사용할 수 있습니다. 

```xml-dtd
allprojects {
    repositories {
        google()
        mavenCentral()
        jcenter() // Warning: this repository is going to shut down soon
        maven {url 'https://jitpack.io'} // Android Draw 라이브러리
    }
}
```

<br>

#### 2.AndroidDraw 라이브러리 의존성 추가

---

그 다음 모듈의 build.gradle 파일에서 아래와 같이 dependencies에 의존성을 추가합니다. 

```xml-dtd
dependencies {

    implementation "org.jetbrains.kotlin:kotlin-stdlib:$kotlin_version"
    implementation 'androidx.core:core-ktx:1.6.0'
    implementation 'androidx.appcompat:appcompat:1.3.1'
    implementation 'com.google.android.material:material:1.4.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.0'

    implementation 'com.github.divyanshub024:AndroidDraw:v0.1' // AndroidDraw 라이브러리 의존성 추가

    testImplementation 'junit:junit:4.+'
    androidTestImplementation 'androidx.test.ext:junit:1.1.3'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.4.0'
}
```

<br>

#### 3. gradle.properties 설정 추가

---

마지막으로 gradle.properties에 다음 두 줄을 추가합니다. 

버전에 따라 이미 있는 경우도 있으므로 없으면 추가합니다. 

```xml-dtd
android.useAndroidX=true
android.enableJetifier=true
```

<br>

❗ **gradle 파일을 수정한 후에는 [Sync Now ] 를 클릭하는 것을 잊지 마세요!**

<br>

#### 4. xml 파일에 DrawView 추가

---

DrawView를 포함한 AndroidDraw의 위젯들은 [Design] 모드에서 바로 사용할 수는 없습니다. 

[Code] 모드에서 **com.divyanshu.draw.widget.DrawView** 태그로 추가해야 합니다. 

다만, 태그만 선언하면 위젯이 [Design] 화면에 생성되기 때문에, 선언만 한 이후에 조작 및 설정은 [Design] 모드에서 하는 것이 편리합니다. 

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".draw.DrawActivity">
		
  	<!-- CrawView 위젯 추가 -->
    <com.divyanshu.draw.widget.DrawView
        android:id="@+id/drawView"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintDimensionRatio="1:1"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.0"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.0" />

    <LinearLayout
        android:id="@+id/linearLayout"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:orientation="horizontal"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@+id/drawView"
        app:layout_constraintVertical_bias="0.316">

        <Button
            android:id="@+id/classifyBtn"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_gravity="center"
            android:layout_marginEnd="10dp"
            android:layout_marginRight="10dp"
            android:layout_weight="0"
            android:gravity="center"
            android:text="CLASSIFY" />

        <Button
            android:id="@+id/clearBtn"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_gravity="center"
            android:layout_marginStart="10dp"
            android:layout_marginLeft="10dp"
            android:layout_weight="0"
            android:width="40dp"
            android:gravity="center"
            android:text="CLEAR" />
    </LinearLayout>

    <TextView
        android:id="@+id/resultView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="RESULT"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.498"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@+id/linearLayout" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

위 xml 코드는 아래의 결과를 만들어냅니다. 

<img src="https://user-images.githubusercontent.com/70505378/129478897-b1d24645-c719-463b-905d-c0932c232c88.png" alt="image-20210815205437837" style="zoom:67%;" />

<br>

<br>

그럼 포스팅 끝!
