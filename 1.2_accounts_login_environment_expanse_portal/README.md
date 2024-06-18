# CIML Summer Institute:   Accounts, Login, Environments, Running Jobs, Logging into Expanse User Portal

## Session: 1.2_accounts_login_environments_running_jobs_expanse_portal

**Date:**  Tuesday, June 18, 2024

**Presented by:** [Mary Thomas](https://www.sdsc.edu/research/researcher_spotlight/thomas_mary.html )  ( mpthomas  @  ucsd.edu ) 

**Title:** Expanse Webinar:  Accessing and Running Jobs on Expanse

### Reading and Presentations:
* **Lecture material:**
   * Presentation Slides: Access, accounts, modules & Envs

     ```
     https://github.com/ciml-org/ciml-summer-institute-2024/blob/main/1.2_accounts_login_environment_expanse_portal/CIML-SI24_Jun_18_2024-PrepDay_accts_login_envs_jobs.pdf
     ```

   * Source Code/Examples:

     ```
     https://github.com/sdsc-hpc-training-org/hpctr-examples
     ```

* **Exercises:**

   * Exercise 1: Log onto Expanse
   Use your training account and password

   ```
   quantum:ciml accountname$ ssh -Y train111@login.expanse.sdsc.edu
   (train111@login.expanse.sdsc.edu) Password: 
   Welcome to Bright release         9.0
                                                         Based on Rocky Linux 8
                                                                    ID: #000002
   --------------------------------------------------------------------------------
                                 WELCOME TO
                  _______  __ ____  ___    _   _______ ______
                 / ____/ |/ // __ \/   |  / | / / ___// ____/
                / __/  |   // /_/ / /| | /  |/ /\__ \/ __/
               / /___ /   |/ ____/ ___ |/ /|  /___/ / /___
              /_____//_/|_/_/   /_/  |_/_/ |_//____/_____/
   --------------------------------------------------------------------------------
   Use the following commands to adjust your environment:
   'module avail'            - show available modules
   'module add <module>'     - adds a module to your environment for this session
   'module initadd <module>' - configure module to be loaded at every login
   -------------------------------------------------------------------------------
   Last login: Mon Jun 17 19:18:38 2024 from 75.80.45.222
   connect /private/tmp/com.apple.launchd.HbagVgBfXZ/org.xquartz:0: Connection refused
   [train111@login02 ~]$ 
   ```

   * Exercise 2: Clone the HPC Training GitHub Repo:
   ```
   [train111@login02 ~]$ git clone https://github.com/sdsc-hpc-training-org/hpctr-examples.git
   Cloning into 'hpctr-examples'...
   remote: Enumerating objects: 485, done.
   remote: Counting objects: 100% (485/485), done.
   remote: Compressing objects: 100% (318/318), done.
   remote: Total 485 (delta 212), reused 421 (delta 158), pack-reused 0
   Receiving objects: 100% (485/485), 27.65 MiB | 27.87 MiB/s, done.
   Resolving deltas: 100% (212/212), done.
   Updating files: 100% (304/304), done.
   [train111@login02 ~]$ ll hpctr-examples/
total 214
drwxr-xr-x 15 train111 gue998    19 Jun 17 22:35 .
drwxr-x--- 10 train111 gue998    20 Jun 17 22:35 ..
drwxr-xr-x  9 train111 gue998    10 Jun 17 22:35 basic_par
drwxr-xr-x  2 train111 gue998     6 Jun 17 22:35 calc-pi
drwxr-xr-x  2 train111 gue998     5 Jun 17 22:35 calc-prime
drwxr-xr-x  6 train111 gue998     7 Jun 17 22:35 cuda
drwxr-xr-x  2 train111 gue998     5 Jun 17 22:35 env_info
-rw-r--r--  1 train111 gue998  5772 Jun 17 22:35 file-tree.txt
drwxr-xr-x  8 train111 gue998    13 Jun 17 22:35 .git
-rw-r--r--  1 train111 gue998  2008 Jun 17 22:35 .gitignore
drwxr-xr-x  2 train111 gue998     3 Jun 17 22:35 gnuplt
drwxr-xr-x  2 train111 gue998     9 Jun 17 22:35 hybrid
-rw-r--r--  1 train111 gue998 35149 Jun 17 22:35 LICENSE
drwxr-xr-x  4 train111 gue998     4 Jun 17 22:35 mkl
drwxr-xr-x  2 train111 gue998    21 Jun 17 22:35 mpi
drwxr-xr-x  2 train111 gue998     3 Jun 17 22:35 netcdf
drwxr-xr-x  2 train111 gue998    17 Jun 17 22:35 openacc
drwxr-xr-x  2 train111 gue998     8 Jun 17 22:35 openmp
-rw-r--r--  1 train111 gue998  5772 Jun 17 22:35 README.md
[train111@login02 ~]$ 

   ```
     
   * Exercise 1: Log onto the Expanse User Portal:
   Use your training account and password

   ```
   https://portal.expanse.sdsc.edu/training
   ```
[Back to Top](#top)
