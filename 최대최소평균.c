#include <stdio.h>

int main(void) {

	int n, i;
	float num;
	float nums[1000]; 
	float max, min; //최대값, 최소값을 넣을 변수
	float avg = 0; //평균값을 넣을 변수
	float sum = 0; //num의 합


	printf("입력할 수가 몇 개인가? :");
	scanf("%d", &n);

	for (i = 0; i < n; i++) {
		printf("수를 입력해주세요 :");
		scanf("%f", &num);
		nums[i] = num;

	}

	max = nums[0];

	for (i = 0; i < n; i++) {
		if (nums[i] >= max) {
			max = nums[i];
		}
	}

	min = nums[0];

	for (i = 0; i < n; i++) {
		if (nums[i] <= min) {
			min = nums[i];
		}
	}

	for (i = 0; i < n; i++) {
		sum += nums[i];
	}

	avg = sum / n;

	printf("최대값: %.2f\n최소값: %.2f\n평균값: %.2f\n", max, min, avg);

}