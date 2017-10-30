package com.example.task5.a8;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.AnalogClock;
import android.widget.DigitalClock;

public class MainActivity extends AppCompatActivity {

    private static Button submit;
    private static DigitalClock digi;
    private static AnalogClock ana;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        myClockSwapApp();
    }

    public void myClockSwapApp() {
//cast your variables:
        digi = (DigitalClock) findViewById(R.id.simpleDigitalClock); //digitalClock here is the id of Digital CLock
        ana = (AnalogClock) findViewById(R.id.simpleAnalogClock); //ana here is the id of Analog
        submit = (Button) findViewById(R.id.button); //button is the id of button
//adding listener to button
        submit.setOnClickListener(new View.OnClickListener(){
                    public void onClick(View v) {
                        
                        if(digi.getVisibility() == DigitalClock.GONE) {
                            digi.setVisibility(DigitalClock.VISIBLE);
                            ana.setVisibility(AnalogClock.GONE);
                        } else {
                            digi.setVisibility(DigitalClock.GONE);
                            ana.setVisibility(AnalogClock. VISIBLE);
                        }

                    }
                }
        );


                Thread td =new Thread(){
                    public void run(){

                        try{
                            sleep(3000);
;
                        }catch(Exception ex){
                            ex.printStacktrace();

                        }finally{
                            Intent it = new Intent(splash.this , MainActivity.class);
                            startActivity(it);
                        }
                    }


                };td.start();
    }

}
