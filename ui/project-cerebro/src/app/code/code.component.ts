import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormGroup, FormControl } from '@angular/forms';
import { MatSnackBar } from "@angular/material/snack-bar";
import { environment } from '../../environments/environment';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-code',
  templateUrl: './code.component.html',
  styleUrls: ['./code.component.css']
})
export class CodeComponent {
  selectedFile = "";
  loading = false;
  fileForm = new FormGroup({
    file: new FormControl(''),
    fileSource: new FormControl('')
  });

  constructor(
    private httpClient: HttpClient,
    private snackBar: MatSnackBar
  ) { }

  onFileChange(event:any) {
    if (event.target.files.length > 0) {
      this.selectedFile = event.target.files[0];
    }
  }

  submit() {
    const baseURL = environment.backendURL;
    const formData = new FormData();
    this.loading = true;
    formData.append('file', this.selectedFile);
    this.httpClient.post(baseURL + "/save-code/ui", formData).subscribe((data:any) => {
        if (data.status == 200)
        {
          this.loading = false;
          this.snackBar.open('Uploaded code files to server!', 'Dismiss', {
            duration: 3000
          });
        } else {
          this.loading = false;
          this.snackBar.open('Error occured', 'Dismiss', {
            duration: 3000
          });
        }
      })
  }
}
