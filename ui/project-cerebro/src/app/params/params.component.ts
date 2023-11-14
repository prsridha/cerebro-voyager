import { Component } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { FormControl, FormGroup } from '@angular/forms';
import { MatSnackBar } from "@angular/material/snack-bar";
import { environment } from '../../environments/environment';

interface ParamsDict {
  train_main: string,
  train_dir: string,
  val_main: string,
  val_dir: string,
  test_main: string,
  test_dir: string,
  misc: [string, string]
}

const httpOptions = {
  headers: new HttpHeaders({
    'Content-Type':  'application/json'
  })
}

@Component({
  selector: 'app-params',
  templateUrl: './params.component.html',
  styleUrls: ['./params.component.css']
})

export class ParamsComponent {
  params = <ParamsDict>{};
  loading = false;
  baseURL = environment.backendURL;
  constructor(
    private httpClient: HttpClient,
    private snackBar: MatSnackBar
    ){}
  
  paramsForm: FormGroup = new FormGroup({
    train_metadata: new FormControl(''),
    train_multimedia: new FormControl(''),
    valid_metadata: new FormControl(''),
    valid_multimedia: new FormControl(''),
    test_metadata: new FormControl(''),
    test_multimedia: new FormControl(''),
    misc_url1: new FormControl(''),
    misc_url2: new FormControl('')
  });

  submitFn(){
    this.params.train_main = this.paramsForm.value.train_metadata
    this.params.train_dir = this.paramsForm.value.train_multimedia
    this.params.val_main = this.paramsForm.value.valid_metadata
    this.params.val_dir = this.paramsForm.value.valid_multimedia
    this.params.test_main =  this.paramsForm.value.test_metadata
    this.params.test_dir = this.paramsForm.value.test_multimedia
    this.params.misc = [this.paramsForm.value.misc_url1, this.paramsForm.value.misc_url2];
    
    this.httpClient.post(this.baseURL + "/params", this.params, httpOptions).subscribe((data: any) => {
      this.loading = true;
      if (data.status == 200)
      {
        this.loading = false;
        this.snackBar.open('Sent parameters to server!', 'Dismiss', {
          duration: 3000
        });
      } else {
        this.loading = false;
        this.snackBar.open('Error occured', 'Dismiss', {
          duration: 3000
        });
      }
    });
  }
}