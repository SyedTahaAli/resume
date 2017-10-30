package com.example.tahaa.task_six;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

public class calculator extends AppCompatActivity {

    EditText edtfirst , edtsecond;
    Button btnadd , btnsub , btnmulti , btnDiv;
    TextView tvResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_calculator);

        edtfirst = (EditText) findViewById(R.id.edtfirst);
        edtsecond = (EditText) findViewById(R.id.edtsecond);

        btnadd = (Button) findViewById(R.id.btnadd);
        btnsub = (Button) findViewById(R.id.btnsub);
        btnmulti = (Button) findViewById(R.id.btnmulti);
        btnDiv = (Button) findViewById(R.id.btnDiv);
        tvResult = (TextView) findViewById(R.id.tvResult);


        btnadd . setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int first , second ,result;
                first = Integer.valueOf(edtfirst.getText().toString());
                second = Integer.valueOf(edtsecond.getText().toString());
                result = first + second;
                tvResult.setText("your result is :"+result);
            }
        });

        btnsub . setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int first , second ,result;
                first = Integer.valueOf(edtfirst.getText().toString());
                second = Integer.valueOf(edtsecond.getText().toString());
                result = first - second;
                tvResult.setText("your result is :"+result);
            }
        });

        btnmulti . setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int first , second ,result;
                first = Integer.valueOf(edtfirst.getText().toString());
                second = Integer.valueOf(edtsecond.getText().toString());
                result = first * second;
                tvResult.setText("your result is :"+result);
            }
        });

        btnDiv . setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int first , second ,result;
                first = Integer.valueOf(edtfirst.getText().toString());
                second = Integer.valueOf(edtsecond.getText().toString());
                result = first / second;
                tvResult.setText("your result is :"+result);
            }
        });







    }
}
