#include <stdio.h>
#include <stdlib.h>
void swap(int* a, int* b)
{
    int temp;
    temp=*a;
    *a=*b;
    *b=temp;

}

void SelectionSort(int* p, int n)
{
    int min,i,j;
    for(i=0;i<n-1;i++)
    {
        min=i;
        for(j=i+1;j<n;j++)
        {
            if(*(p+min)>*(p+j))
            {
                min=j;
            }
        }
        swap(p+i,p+min);
    }
}
void BubbleSort(int* p, int n)
{
    int k,j,flag;
    for(j=1;j<=n-1;j++)
    {
        flag=0;
        for(k=0;k<n-j;k++)
        {
            if(*(p+k)>*(p+k+1))
            {
                swap(p+k,p+k+1);
                flag=1;
            }
        }
        if(flag==0)
        {
            break;
        }
    }
}
void InsertionSort(int*p,int n)
{
    int i,j,temp;
    for(i=1;i<=n-1;i++)
    {
        temp=i;
        for(j=i-1;j>=0;j--)
        {
            if(*(p+j)>*(p+temp))
            {
                swap(p+j,p+temp);
                temp--;
            }
            else
                break;
        }
    }
}
int Partition(int a[], int start, int end)
{
    int pivot,pindex,i=0;
    if(start<end)
    {
        pivot=a[end];
        pindex=start;
        for(i=start;i<=end-1;i++)
        {
            if(a[i]<pivot)
            {
                swap(&a[i],&a[pindex]);
                pindex++;
            }
        }
        swap(&a[end],&a[pindex]);
        return(pindex);
    }
}
void QuickSort(int a[], int start, int end)
{   int pindex;
    if(start<end)
    {
        pindex=Partition(a, start,end);
        QuickSort(a, start, pindex-1);
        QuickSort(a, pindex+1, end);
    }

}
void CountingSort(int *p, int n, int max)
{
   int count[max+1],i,j,k;
   for(i=0;i<max+1;i++)
   {
       count[i]=0;
   }

   for(i=0;i<n;i++)
   {
       count[*(p+i)]=count[*(p+i)]+1;
   }
   for(i=0;i<max+1;i++)
   {
       for(j=0;j<count[i];j++)
       {
          *p=i;
            p++;
       }
   }
}
int GetMax(int a[], int n)
{
    int i=0,max;
    max=a[0];
    for(i=1;i<n;i++)
    {
        if(max<a[i])
        {
            max=a[i];
        }
    }
    return max;
}
void RedixCountingSort(int *p, int n, int div)
{
    int count[10],i,j;
    for(i=0;i<n;i++)
    {
        count[(p[i]/div)%10]++;

    }
    for(i=1;i<10;i++)
    {
        count[i]+=count[i-1];
    }
    int output[n];
    for(i=n-1;i>0;i--)
    {
        output[count[(p[i]/div)%10]]=p[i];
        count[(p[i]/div)%10]--;
    }
    for(i=0;i<=n;i++)
    {
      p[i]=output[i];
    }

}
void RadixSort(int a[],int n)
{
    int max,i, div;
    max=GetMax(a, n);
    for(div=1;(max/div)!=0;div*=10)
    {
        RedixCountingSort(a,n,div);
        for(i=0;i<n;i++)
        {
            printf("%d \n",a[i]);
        }

    }
}
void Merge(int a[], int l,int m, int r )
{
    int n1,n2;
    n1=m-l+1;
    n2=r-m;
    int L[n1];
    int R[n2];
    int i=0,j=0,k=l;
    for(i=0;i<n1;i++)
    {
        L[i]=a[l+i];
    }
    for(i=0;i<n2;i++)
    {
        R[i]=a[m+1+i];
    }
    i=0;
    j=0;
    while(i<n1&&j<n2)
    {
        if(L[i]<=R[j])
        {
            a[k]=L[i];
            i++;
            k++;
        }
        else
        {
            a[k]=R[j];
            j++;
            k++;
        }
    }
    while(i<n1)
    {
        a[k]=L[i];
        i++;
        k++;
    }
    while(j<n2)
    {
        a[k]=R[j];
        j++;
        k++;
    }
}
void MergeSort(int a[], int l, int r)
{
    int m;
    if(r>l)
     {
        m=l+(r-l)/2;
        MergeSort(a,l,m);
        MergeSort(a,m+1,r);
        Merge(a,l,m,r);
     }
}
int main()
{
    int i,n;
    printf("Enter size of array \n");
    scanf("%d",&n);
    printf("Enter Number \n");
    int a[n];
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    MergeSort(a, 0 ,n-1);
    printf("Sorted array is...");
    for(i=0;i<n;i++)
    {
        printf("%d ",a[i]);
    }
}
