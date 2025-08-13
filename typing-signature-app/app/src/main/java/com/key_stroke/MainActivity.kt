package com.hyunsung.key_stroke

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.background
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Settings
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.shape.RoundedCornerShape
import com.hyunsung.key_stroke.data.Constants
import com.hyunsung.key_stroke.ui.theme.KeystrokeTheme
import android.util.Log
import com.hyunsung.key_stroke.ui.KeystrokeActivity

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("MainActivity", "MainActivity created")
        setContent {
            KeystrokeTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                                    LoginScreen(
                    onLogin = { userId ->
                        navigateToKeystroke(userId)
                    }
                )
                }
            }
        }
    }

    private fun navigateToKeystroke(loginId: String) {
        Log.d("MainActivity", "Navigating to KeystrokeActivity with loginId: $loginId")
        val intent = Intent(this, KeystrokeActivity::class.java).apply {
            putExtra(Constants.EXTRA_LOGIN_ID, loginId)
        }
        startActivity(intent)
    }
}

@Composable
fun LoginScreen(onLogin: (String) -> Unit) {
    var username by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var selectedTab by remember { mutableStateOf(1) } // 0: 공동·금융인증서, 1: 아이디 로그인
    
    fun logLoginButtonClick() {
        Log.d("LoginScreen", "Login button clicked. Username: $username")
    }
    fun logFindIdClick() {
        Log.d("LoginScreen", "Find ID button clicked")
    }
    fun logFindPwClick() {
        Log.d("LoginScreen", "Find Password button clicked")
    }
    fun logSignUpClick() {
        Log.d("LoginScreen", "Sign Up button clicked")
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFFEF9C3)),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // 헤더 섹션
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color(0xFFFFFCE8)
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                // 상단 로고 및 제목
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Card(
                        modifier = Modifier.size(32.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = Color(0xFFF59E0B)
                        ),
                        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
                    ) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "KS",
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.width(8.dp))
                    
                    Text(
                        text = "타이핑 서명 로그인",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = Color(0xFF1F2937)
                    )
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // 탭 버튼들
                Row(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    // 공동·금융인증서 탭
                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .height(36.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = if (selectedTab == 0) Color(0xFFFEF9C3) else Color(0xFFF3F4F6)
                        ),
                        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
                    ) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "공동·금융인증서",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Medium,
                                color = Color(0xFF1F2937)
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.width(8.dp))
                    
                    // 아이디 로그인 탭
                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .height(36.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = if (selectedTab == 1) Color(0xFFFEF9C3) else Color(0xFFF3F4F6)
                        ),
                        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
                    ) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "아이디 로그인",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Medium,
                                color = Color(0xFF1F2937)
                            )
                        }
                    }
                }
            }
        }
        
        // 메인 로그인 폼
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color.White
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp)
            ) {
                // 아이디 입력
                Column(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = "아이디",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium,
                        color = Color(0xFF1F2937),
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    
                    OutlinedTextField(
                        value = username,
                        onValueChange = { username = it },
                        placeholder = { Text("아이디") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Text,
                            imeAction = ImeAction.Next
                        ),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Color(0xFFF59E0B),
                            unfocusedBorderColor = Color(0xFFE5E7EB),
                            focusedContainerColor = Color.White,
                            unfocusedContainerColor = Color.White
                        ),
                        textStyle = androidx.compose.ui.text.TextStyle(
                            fontSize = 16.sp,
                            color = Color(0xFF1F2937)
                        )
                    )
                }
                
                // 비밀번호 입력
                Column(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = "비밀번호",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium,
                        color = Color(0xFF1F2937),
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    
                    OutlinedTextField(
                        value = password,
                        onValueChange = { password = it },
                        placeholder = { Text("비밀번호") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                        visualTransformation = PasswordVisualTransformation(),
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Password,
                            imeAction = ImeAction.Done
                        ),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Color(0xFFF59E0B),
                            unfocusedBorderColor = Color(0xFFE5E7EB),
                            focusedContainerColor = Color.White,
                            unfocusedContainerColor = Color.White
                        ),
                        textStyle = androidx.compose.ui.text.TextStyle(
                            fontSize = 16.sp,
                            color = Color(0xFF1F2937)
                        )
                    )
                }
                
                // 로그인 버튼
                Button(
                    onClick = { 
                        if (username.isNotBlank() && password.isNotBlank()) {
                            logLoginButtonClick()
                            onLogin(username)
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFFEF9C3),
                        contentColor = Color(0xFF1F2937)
                    ),
                    enabled = username.isNotBlank() && password.isNotBlank(),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        "로그인",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                
                // 하단 링크들
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextButton(
                        onClick = { 
                            logFindIdClick()
                            onLogin("verify") 
                        },
                        colors = ButtonDefaults.textButtonColors(
                            contentColor = Color(0xFF6B7280)
                        )
                    ) {
                        Text("아이디 조회", fontSize = 13.sp)
                    }
                    
                    Text("|", color = Color(0xFFD1D5DB), fontSize = 13.sp)
                    
                    TextButton(
                        onClick = { 
                            logFindPwClick()
                            onLogin("verify") 
                        },
                        colors = ButtonDefaults.textButtonColors(
                            contentColor = Color(0xFF6B7280)
                        )
                    ) {
                        Text("비밀번호 찾기", fontSize = 13.sp)
                    }
                    
                    Text("|", color = Color(0xFF6B7280), fontSize = 13.sp)
                    
                    TextButton(
                        onClick = { 
                            logSignUpClick()
                            onLogin("verify") 
                        },
                        colors = ButtonDefaults.textButtonColors(
                            contentColor = Color(0xFF6B7280)
                        )
                    ) {
                        Text("회원가입", fontSize = 13.sp)
                    }
                }
            }
        }
        
        // 하단 네비게이션
        Spacer(modifier = Modifier.weight(1f))
        
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color(0xFFF3F4F6)
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                // KS타이핑서명인증서 발급
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Card(
                        modifier = Modifier.size(24.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = Color(0xFFF59E0B)
                        ),
                        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
                    ) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "KS",
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                        }
                    }
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    Text(
                        text = "인증서 발급",
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center,
                        color = Color(0xFF1F2937)
                    )
                }
                
                // 인증센터
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = "인증센터",
                        modifier = Modifier.size(24.dp),
                        tint = Color(0xFF1F2937)
                    )
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    Text(
                        text = "인증센터",
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center,
                        color = Color(0xFF1F2937)
                    )
                }
                
                // 로그인 설정
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.Settings,
                        contentDescription = "로그인 설정",
                        modifier = Modifier.size(24.dp),
                        tint = Color(0xFF1F2937)
                    )
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    Text(
                        text = "로그인 설정",
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center,
                        color = Color(0xFF1F2937)
                    )
                }
                
                // 인증서 이용 안내
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.Info,
                        contentDescription = "인증서 안내",
                        modifier = Modifier.size(24.dp),
                        tint = Color(0xFF1F2937)
                    )
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    Text(
                        text = "인증서 이용\n안내",
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center,
                        color = Color(0xFF1F2937)
                    )
                }
            }
        }
    }
}
